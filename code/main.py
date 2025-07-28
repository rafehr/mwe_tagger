import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizerFast
from sklearn.model_selection import RepeatedKFold
import numpy as np

from data import (
    StreusleDataset, # type: ignore
    ParsemeDataset, # type: ignore
    read_streusle_conllulex, # type: ignore
    read_parseme_cupt, # type: ignore
    collate_fn, # type: ignore
    get_label_dict, # type: ignore
    get_deprel_dict, # type: ignore
    create_subset # type: ignore
) 
from preprocessing import change_lextag_labels, change_deprels
from model import MWETagger, MWETaggerDep
from train import train

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('config_path', help='Path to config file')
args = arg_parser.parse_args()

# Read the config file
with open(args.config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Data configs
TRAIN_PATH = Path(config['data']['train_path'])
DEV_PATH = Path(config['data']['dev_path'])
TEST_PATH = Path(config['data']['test_path'])
BIO_SCHEME = config['data']['bio_scheme']
# # DATA_SET_TYPE = config['data']['data_set_type']

# Model configs
PRETRAINED_MODEL_NAME = config['model']['pretrained_model_name']
TOKENIZER_NAME = config['model']['tokenizer_name']
ADD_DEP_EMBS = config['model']['add_dep_embs']

# Training configs
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
SAVE_DIR = Path(config['training']['save_dir'])
PATIENCE = config['training']['patience']
CROSS_VAL = config['training']['cross_validation']
SAVE_PREDICTIONS = config['training']['save_predictions']

# Read STREUSLE data
train_sents = read_parseme_cupt(TRAIN_PATH)
dev_sents = read_parseme_cupt(DEV_PATH)
test_sents = read_parseme_cupt(TEST_PATH)

# Create data sets
train_data = ParsemeDataset(train_sents)
dev_data = ParsemeDataset(dev_sents)
test_data = ParsemeDataset(test_sents)


# Change LEXTAG labels so that only VMWEs have IOB labels (including the
# vmwe category, i.e. B-VID) and everything else receives the 'O' tag
# change_lextag_labels(train_data.sents + dev_data.sents + test_data.sents)

# change_deprels(train_data.sents)


# Specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
print(f"Using the following device: {device_name}")
print(f"Cross validation: {CROSS_VAL}")

# Fetch the BIO-style labels that include MWE information and create
# a label dictionary that includes all labels (train, dev and test).
all_sents = train_data.sents + dev_data.sents + test_data.sents
label_to_id, id_to_label = get_label_dict(
    data=all_sents,
    streusle=False
)
print(f"Using the following labels: {label_to_id}")

# Fetch dependency relations and create dictionaries that map them
# to integers and vice versa
deprel_to_id, id_to_deprel = get_deprel_dict(data=all_sents)
print(deprel_to_id)

# Instantiate BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)

if not CROSS_VAL:
    # Create data loaders for train and dev
    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            label_to_id=label_to_id,
            deprel_to_id=deprel_to_id,
            tokenizer=tokenizer,
            max_len=128
        )
    )

    dev_data_loader = DataLoader(
        dataset=dev_data,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            label_to_id=label_to_id,
            deprel_to_id=deprel_to_id,
            tokenizer=tokenizer,
            max_len=128
        )
    )

    # Instantiate the model
    model = MWETaggerDep(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        num_labels=len(label_to_id),
        num_deprels=len(deprel_to_id),
        deprel_emb_dim=64,
        device=device
    ).to(device)

    print(f"Using the following model: \n{model}")

    # Train the model
    best_model_eval_metrics = train(
        model=model,
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        train_data_loader=train_data_loader,
        dev_data_loader=dev_data_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        tokenizer=tokenizer,
        add_deprels=ADD_DEP_EMBS,
        save_dir=SAVE_DIR
    )
else:
    print("Performing cross validation")
    rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    # Add dev and test data to train_data. It is necessary to access
    # the sentences in the StreusleDataset object because only there
    # the labels are changed. I.e. train_data.sents instead of train_sents 
    all_data = StreusleDataset(
        sents=train_data.sents + dev_data.sents + test_data.sents
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pretrained_model_name = PRETRAINED_MODEL_NAME.replace('/', '_')
    results_dir = SAVE_DIR / 'results' / f'run_{pretrained_model_name}_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'class_reports': [],
        'mean_f1_score': 0,
        'mean_precision_score': 0,
        'mean_recall_score': 0,
        'std_f1_score': 0,
        'std_precision_score': 0,
        'std_recall_score': 0
    }

    for fold, (train_idxs, val_idxs) in enumerate(rkf.split(all_data)):
        print('****************************************************')
        print(f'Fold: {fold}')
        print('****************************************************')
        train_subset = create_subset(all_data, train_idxs.tolist())
        val_subset = create_subset(all_data, val_idxs.tolist())

        train_data_loader = DataLoader(
            dataset=train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(
                batch=batch,
                label_to_id=label_to_id,
                deprel_to_id=deprel_to_id,
                tokenizer=tokenizer,
                max_len=128
            )
        )

        val_data_loader = DataLoader(
            dataset=val_subset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(
                batch=batch,
                label_to_id=label_to_id,
                deprel_to_id=deprel_to_id,
                tokenizer=tokenizer,
                max_len=128
            )
        )

        if ADD_DEP_EMBS:
            # Instantiate the model with dependency embeddings
            model = MWETaggerDep(
                pretrained_model_name=PRETRAINED_MODEL_NAME,
                num_labels=len(label_to_id),
                num_deprels=len(deprel_to_id),
                deprel_emb_dim=64,
                device=device
            ).to(device)
        else:
            # Instantiate the model
            model = MWETagger(
                pretrained_model_name=PRETRAINED_MODEL_NAME,
                num_labels=len(label_to_id),
                device=device
            ).to(device)
            
        print(f"Using the following model: \n{model}")

        # Train the model
        best_model_eval_metrics = train(
            model=model,
            pretrained_model_name=PRETRAINED_MODEL_NAME,
            train_data_loader=train_data_loader,
            dev_data_loader=val_data_loader,
            device=device,
            num_epochs=NUM_EPOCHS,
            patience=PATIENCE,
            learning_rate=LEARNING_RATE,
            add_deprels=ADD_DEP_EMBS,
            tokenizer=tokenizer
        )


        # Creating the prediction dict and filling it with all the
        # information needed for error analysis
        preds_dict = {}

        gold_labels = best_model_eval_metrics.gold_labels
        preds = best_model_eval_metrics.predictions
        sent_ids = best_model_eval_metrics.sent_ids
        og_sents = best_model_eval_metrics.og_sents

        for gl, p, sent_idx, og_sent in zip(gold_labels, preds, sent_ids, og_sents): # type: ignore
            preds_dict[sent_idx] = {
                'sentence': og_sent,
                'predictions': p,
                'gold_labels': gl,
                'preds_vs_gold': [(pr, go) for pr, go in zip(p, gl)]
            }
        

        with open(results_dir / f'pred_dict_{fold}.json', 'w') as f:
            json.dump(preds_dict, f, indent=4)

        results_dict['f1_scores'].append(
            best_model_eval_metrics.f1
        )
        results_dict['precision_scores'].append(
            best_model_eval_metrics.precision
        )
        results_dict['recall_scores'].append(
            best_model_eval_metrics.recall
        )
        # results_dict['class_reports'].append(
        #     best_model_eval_metrics.classification_report
        # )

    results_dict['mean_f1_score'] = np.mean(
        results_dict['f1_scores']
    )
    results_dict['mean_precision_score'] = np.mean(
        results_dict['precision_scores']
    )
    results_dict['mean_recall_score'] = np.mean(
        results_dict['recall_scores']
    )
    results_dict['std_f1_score'] = np.std(
        results_dict['f1_scores']
    )
    results_dict['std_precision_score'] = np.std(
        results_dict['precision_scores']
    )
    results_dict['std_recall_score'] = np.std(
        results_dict['recall_scores']
    )

    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

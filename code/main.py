import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizerFast
from sklearn.model_selection import KFold
import numpy as np

from data import StreusleDataset, read_streusle_conllulex, collate_fn, get_label_dict, create_subset # type: ignore
from preprocessing import change_lextag_labels
from model import MWETagger
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

# Model configs
PRETRAINED_MODEL_NAME = config['model']['pretrained_model_name']
TOKENIZER_NAME = config['model']['tokenizer_name']

# Training configs
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
SAVE_DIR = Path(config['training']['save_dir'])
PATIENCE = config['training']['patience']
CROSS_VAL = config['training']['cross_validation']

# Read STREUSLE data
train_sents = read_streusle_conllulex(TRAIN_PATH)
dev_sents = read_streusle_conllulex(DEV_PATH)
test_sents = read_streusle_conllulex(TEST_PATH)

# Create data sets
train_data = StreusleDataset(train_sents)
dev_data = StreusleDataset(dev_sents)
test_data = StreusleDataset(test_sents)

# Change LEXTAG labels so that only VMWEs have IOB labels (including the
# vmwe category, i.e. B-VID) and everything else receives the 'O' tag
change_lextag_labels(train_data.sents + dev_data.sents + test_data.sents)

# Specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
print(f"Using the following device: {device_name}")
print(f"Cross validation: {CROSS_VAL}")

# Fetch the BIO-style labels that include MWE information and create
# a label dictionary that includes all labels (train, dev and test).
all_sents = train_data.sents + dev_data.sents + test_data.sents
label_to_id, id_to_label = get_label_dict(data=all_sents)
print(f"Using the following labels: {label_to_id}")

# Instantiate BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)

if not CROSS_VAL:
    print(len(train_data))
    print(len(dev_data))
    # Create data loaders for train and dev
    train_data_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(
            batch=batch,
            label_to_id=label_to_id,
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
            tokenizer=tokenizer,
            max_len=128
        )
    )

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
        dev_data_loader=dev_data_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        save_dir=SAVE_DIR
    )
else:
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Add dev and test data to train_data. It is necessary to access
    # the sentences in the StreusleDataset object because only there
    # the labels are changed. I.e. train_data.sents instead of train_sents 
    all_data = StreusleDataset(
        sents=train_data.sents + dev_data.sents + test_data.sents
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = SAVE_DIR / f'cross_val_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'class_reports': [],
        'mean_f1_score': 0,
        'mean_precision_score': 0,
        'mean_recall_score': 0
    }

    for train_idxs, val_idxs in kf.split(all_data):
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
                tokenizer=tokenizer,
                max_len=128
            )
        )

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
            learning_rate=LEARNING_RATE
        )

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
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

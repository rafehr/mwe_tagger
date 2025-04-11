import json
import argparse
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from seqeval.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, classification_report
)

from data import StreusleDataset, collate_fn, read_streusle_conllulex # type: ignore
from preprocessing import change_lextag_labels
from model import MWETagger

@dataclass
class EvalMetrics:
    accuracy: float | List[float]
    precision: float | List[float]
    recall: float | List[float] 
    f1: float | List[float] | None
    classification_report: str | Dict[Any, Any]


def remove_ignore_labels(
    gold_labels: List[List[int]],
    predictions: List[List[int]]
) -> Tuple[List[List[int]], List[List[int]]]:
    """Removes the placeholder -100 and the corresponding predictions."""
    true_labels = []
    true_predictions = []
    for label, pred in zip(gold_labels, predictions):
        true_l = []
        true_p = []
        for l, p in zip(label, pred):
            if l != -100:
                true_l.append(l)
                true_p.append(p)
        assert len(true_l) == len(true_p)
        true_labels.append(true_l)
        true_predictions.append(true_p)
    return true_labels, true_predictions


def convert_labels(
    gold_labels: List[List[int]],
    predictions: List[List[int]],
    label_path: str
) -> Tuple[List[List[str]], List[List[str]]]:
    """Converts integers labes back to BIO-style labels."""
    with open(label_path, 'r') as f:
        id_to_label = json.load(f)
    id_to_label = {int(k): v for k, v in id_to_label.items()}
        
    gold_conv_labels = []
    for labels in gold_labels:
        gold_conv_labels.append([id_to_label[l] for l in labels])
    
    conv_predictions = []
    for pred in predictions:
        conv_predictions.append([id_to_label[l] for l in pred])
    return gold_conv_labels, conv_predictions


def compute_eval_metrics(
    gold_labels: List[List[int]],
    preds: List[List[int]],
    label_path: str
) -> EvalMetrics: 
    """Computs accuracy and F1 with seqeval.  """
    # Remove -100 and the corresponding predictions
    gold_labels, preds = remove_ignore_labels(gold_labels, preds)
    # Convert integer labels back to IOB labels
    gold_conv_labels, conv_predictions = convert_labels(
        gold_labels=gold_labels,
        predictions=preds,
        label_path=label_path
    )
    accuracy = accuracy_score(gold_conv_labels, conv_predictions)
    precision = precision_score(gold_conv_labels, conv_predictions)
    recall = recall_score(gold_conv_labels, conv_predictions)
    f1 = f1_score(gold_conv_labels, conv_predictions)
    class_report = classification_report(
        gold_conv_labels,
        conv_predictions,
        output_dict=True
    )
    return EvalMetrics(accuracy, precision, recall, f1, class_report)


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: str,
    batch_size: int
) -> Tuple[float, EvalMetrics]: 
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        for batch in tqdm(data_loader):
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            # Get batch size for weighing the batch-wise loss
            batch_size = batch['input_ids'].shape[0]
            # Make predictions
            logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            predictions = torch.argmax(logits, dim=2)
            
            # Compute the loss
            loss = criterion(
                    logits.view(-1, logits.shape[2]),
                    batch_labels.view(-1)
            )
                    
            # Add batch-wise predictions and labels to overall
            # predictions and labels
            all_predictions.extend(predictions.tolist())
            all_labels.extend(batch_labels.tolist())
            
            # Add batch loss to total loss and weigh it by batch size
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        average_loss = total_loss/total_samples
        print(f"Loss: {average_loss:.4f}")


        eval_metrics = compute_eval_metrics(
            gold_labels=all_labels,
            preds=all_predictions,
            label_path='id_to_label.json'
        )
    return average_loss, eval_metrics
 
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('config_path', help='Path to config file')
    arg_parser.add_argument('model_path', help='Path to trained model')
    arg_parser.add_argument(
        'data_path',
        help='Path to the data we want the model to be evaluated on.'
    )
    args = arg_parser.parse_args()

    # Read the config file
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Configs
    PRETRAINED_MODEL_NAME = config['model']['pretrained_model_name']
    TOKENIZER_NAME = config['model']['tokenizer_name']
    BATCH_SIZE = config['training']['batch_size']

    # Specify device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
    print(f"Using the following device: {device_name}")

    # Reade STREUSLE data
    sents = read_streusle_conllulex(args.data_path)

    # Create data set
    data = StreusleDataset(sents)
 
    # Change LEXTAG labels so that only VMWEs have IOB labels (including the
    # vmwe category, i.e. B-VID) and everything else receives the 'O' tag
    change_lextag_labels(data.sents)

    # Instantiate BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)

    # Load mapping from label to id
    with open('label_to_id.json') as f:
        label_to_id = json.load(f)

    # Create data loaders for train and dev
    data_loader = DataLoader(
        dataset=data,
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

    # Instantiate the model
    model = MWETagger(
        pretrained_model_name=PRETRAINED_MODEL_NAME,
        num_labels=len(label_to_id),
        device=device
    ).to(device)

    # Load trained models state dict
    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=torch.device(device)
        )
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    val_loss, eval_metrics = evaluate(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        batch_size=BATCH_SIZE
    )

    print(f"F1-Score: {eval_metrics.f1}")
    print("Classification Report:")
    print(eval_metrics.classification_report)


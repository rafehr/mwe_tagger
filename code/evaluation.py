import json
import argparse
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from seqeval.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, classification_report
)

from data import StreusleDataset, collate_fn # type: ignore
from preprocessing import change_lextag_labels
from model import EnsembleMWETagger

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
    """Converts integers labels back to BIO-style labels."""
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

def fetch_majority_vote(
    logits_base: torch.Tensor,
    logits_sem: torch.Tensor,
    logits_syn: torch.Tensor
) -> torch.Tensor:
    """Determines the majority vote for three different MWE taggers.
    If there is a tie, the vote with the "most confidence" (i.e.
    the largest logit value) is chosen.

    Args:
        logits_base: Logits of the baseline model
        logits_sem: Logits of the semantically enhanced model
        logits_syn: Logits of the syntactically enhanced model

    Returns:
        final_votes: The final predictions
    """
    max_values_base, val_idxs_base = torch.max(logits_base, dim=2)
    max_values_sem, val_idxs_sem = torch.max(logits_sem, dim=2)
    max_values_syn, val_idxs_syn = torch.max(logits_syn, dim=2)
    
    all_max_values = torch.stack(
        (max_values_base, max_values_sem, max_values_syn)
    )
    all_idxs = torch.stack(
        (val_idxs_base, val_idxs_sem, val_idxs_syn)
    )
    _, val_idxs_per_tok = torch.max(all_max_values, dim=0)
        
    num_rows = max_values_base.shape[0]
    num_columns = max_values_base.shape[1]
    final_votes = torch.zeros(num_rows, num_columns, dtype=torch.long)
        
    for i in range(0, num_rows):
        for j in range(0, num_columns):
            votes = []
            votes.append(val_idxs_base[i, j].item())
            votes.append(val_idxs_sem[i, j].item())
            votes.append(val_idxs_syn[i, j].item())
            most_common = Counter(votes).most_common()
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                final_votes[i][j] = most_common[0][0]
            else:
                final_votes[i][j] = all_idxs[val_idxs_per_tok[i][j]][i][j]
    return final_votes


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
            logits_base, logits_sem, logits_syn = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
            logits_final = fetch_majority_vote(
                logits_base=logits_base,
                logits_sem=logits_sem,
                logits_syn=logits_syn
            )
            predictions = torch.argmax(logits_final, dim=2)
            
            # Compute the loss
            loss = criterion(
                    logits_final.view(-1, logits_final.shape[2]),
                    batch_labels.view(-1)
            )
            exit()
             
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
    PRETRAINED_MODEL_NAME_BASE = config['model']['pretrained_model_name_base']
    PRETRAINED_MODEL_NAME_SEM = config['model']['pretrained_model_name_sem']
    PRETRAINED_MODEL_NAME_SYN = config['model']['pretrained_model_name_syn']
    TOKENIZER_NAME = config['model']['tokenizer_name']
    BATCH_SIZE = config['training']['batch_size']

    # Specify device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
    print(f"Using the following device: {device_name}")

    # Read STREUSLE data and create data sets
    data = StreusleDataset(args.data_path)
 
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
    model = EnsembleMWETagger(
        pretrained_model_name_base=PRETRAINED_MODEL_NAME_BASE,
        pretrained_model_name_syn=PRETRAINED_MODEL_NAME_SYN,
        pretrained_model_name_sem=PRETRAINED_MODEL_NAME_SEM,
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


import json
from typing import Tuple, List

import torch
import torch.nn as nn
from tqdm import tqdm

from seqeval.metrics import f1_score, accuracy_score, classification_report

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
):
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
    f1 = f1_score(gold_conv_labels, conv_predictions)
    return accuracy, f1


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: str,
    batch_size: int
):
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


        accuracy, f1 = compute_eval_metrics(
            gold_labels=all_labels,
            preds=all_predictions,
            label_path='id_to_label.json'
        )
        print(f"Accuracy: {accuracy}, F1-Score: {f1}")
        # print(f"Accuracy validation set: {accuracy}")
    return loss
 

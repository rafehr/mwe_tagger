import os
from typing import Optional
from pathlib import Path
from datetime import datetime

from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from evaluation import evaluate, EvalMetrics
from data import save_train_metadata # type: ignore

def train(
    model: nn.Module,
    pretrained_model_name: str,
    train_data_loader: torch.utils.data.DataLoader,
    dev_data_loader: torch.utils.data.DataLoader,
    device: str,
    num_epochs: int,
    patience: int,
    learning_rate: float,
    save_dir: Optional[Path] = None
) -> EvalMetrics:
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    model.train()

    # Define criteria for early stopping
    best_val_loss = float('inf')
    counter = 0
    early_stopping_triggered = False

    best_val_f1 = None

    # Generate timestamp to give a unique name to the directory
    # the model is saved in
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_dir:
        pretrained_model_name = pretrained_model_name.replace('/', '_')
        model_dir = save_dir / f'run_{pretrained_model_name}_{timestamp}'
        model_dir.mkdir(parents=True, exist_ok=True)

    # Train loop
    for epoch in trange(num_epochs, desc='Epoch'):
        total_loss = 0
        total_samples = 0
        # Making batch-wise predictions
        for batch in tqdm(train_data_loader):
            batch_input_ids = batch['input_ids'].to(device)
            batch_attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)

            # Get batch size for weighing the batch-wise loss
            batch_size = batch['input_ids'].shape[0]
            # Make predictions 
            logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )

            # Compute the loss
            loss = criterion(
                logits.view(-1, logits.shape[2]),
                batch_labels.view(-1)
            )

            # Add batch loss to total loss and weigh it by batch size
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute the gradients
            loss.backward()
            # Update the weights
            optimizer.step()
            # Reset the gradients
            optimizer.zero_grad()
            # break

        # Print loss averaged over all batches 
        average_loss = total_loss/total_samples
        print(f"{'-' * 40} TRAINING LOSS {'-' * 40}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

        # Evaluate the model on the dev set
        print(f"{'-' * 30} VALIDATION LOSS {'-' * 30}")
        val_loss, eval_metrics = evaluate(
            model=model,
            data_loader=dev_data_loader,
            criterion=criterion,
            device=device,
            batch_size=batch_size
        )

        print(f"F1-Score: {eval_metrics.f1}")

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = eval_metrics.f1
            best_val_accuracy = eval_metrics.accuracy
            best_val_precision = eval_metrics.precision
            best_val_recall = eval_metrics.recall
            best_val_class_report = eval_metrics.classification_report
            counter = 0
            if save_dir:
                torch.save(
                    model.state_dict(),
                    model_dir / 'best_model.pth' # type: ignore
                ) 
        else:
            counter += 1
            if counter >= patience:
                early_stopping_triggered = True
                print("Early stopping triggered. Stopping training.")
                break

        # Put model back into training mode
        model.train()
        # break

    if save_dir:
        save_train_metadata(
            num_epochs=epoch,
            best_val_f1_score=best_val_f1,
            early_stopping_triggered=early_stopping_triggered,
            save_dir=model_dir
        )
    return EvalMetrics(
        best_val_accuracy,
        best_val_precision,
        best_val_recall,
        best_val_f1,
        best_val_class_report
    )
    

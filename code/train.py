import os
from pathlib import Path

from tqdm import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from evaluation import evaluate

def train(
    model: nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    dev_data_loader: torch.utils.data.DataLoader,
    device: str,
    num_epochs: int,
    learning_rate: float,
    save_dir: Path
):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    # Define criteria for early stopping
    best_val_loss = float('inf')
    patience = 2
    counter = 0

    if not save_dir.exists():
        os.makedirs(save_dir)

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
            break

        # Print loss averaged over all batches 
        average_loss = total_loss/total_samples
        print(f"{'-' * 40} TRAINING LOSS {'-' * 40}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

        # Evaluate the model on the dev set
        print(f"{'-' * 30} VALIDATION LOSS {'-' * 30}")
        val_loss = evaluate(
            model=model,
            data_loader=dev_data_loader,
            criterion=criterion,
            batch_size=batch_size
        )

        # Save model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_model.pth")
            ) 
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break

        # Put model back into training mode
        model.train()
        exit()
        

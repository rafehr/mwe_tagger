from tqdm import trange, tqdm
import torch
import torch.nn as nn

def train(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    num_epochs: int
):
    model.train()
    for epoch in trange(num_epochs, desc='Epoch'):
        for step, batch in enumerate(tqdm(data_loader)):
            logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            print(batch['labels'])
            # print(logits.shape)
            exit()
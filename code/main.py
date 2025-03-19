import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import BertTokenizerFast

from data import StreusleDataset, collate_fn, get_label_dict # type: ignore
from preprocessing import change_lextag_labels
from model import MWETagger
from train import train

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('config_path')
args = arg_parser.parse_args()

# Read the config file
with open(args.config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Data configs
TRAIN_PATH = Path(config['data']['train_path'])
DEV_PATH = Path(config['data']['dev_path'])
TEST_PATH = Path(config['data']['test_path'])

# Model configs
MODEL_NAME = config["model"]["model_name"]

# Training configs
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
LEARNING_RATE = config['training']['learning_rate']
SAVE_DIR = Path(config['training']['save_dir'])
BIO_SCHEME = config['data']['bio_scheme']

# Read STREUSLE data and create data sets
train_data = StreusleDataset(TRAIN_PATH)
dev_data = StreusleDataset(DEV_PATH)
test_data = StreusleDataset(TEST_PATH)


# Change LEXTAG labels so that only VMWEs have IOB labels (including the
# vmwe category, i.e. B-VID) and everything else receives the 'O' tag
change_lextag_labels(train_data.sents + dev_data.sents + test_data.sents)

# Specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
print(f"Using the following device: {device_name}")

# Fetch the BIO-style labels that include MWE information and create
# a label dictionary that includes all labels (train, dev and test).
label_to_id, id_to_label = get_label_dict(
    data = train_data.sents + dev_data.sents + test_data.sents
)

# Instantiate BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

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
    model_name=MODEL_NAME,
    num_labels=len(label_to_id)
).to(device)

# Train the model
train(
    model=model,
    train_data_loader=train_data_loader,
    dev_data_loader=dev_data_loader,
    device=device,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    save_dir=SAVE_DIR
)

# Evaluate the model

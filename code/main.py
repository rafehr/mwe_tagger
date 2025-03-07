import argparse
import json

from tqdm import trange, tqdm
import seqeval
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import BertTokenizerFast

from data import StreusleDataset, collate_fn # type: ignore
from model import MWETagger

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('config_path')
args = arg_parser.parse_args()

# Read the config file
with open(args.config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# Data
TRAIN_PATH = config['data']['train_path']
DEV_PATH = config['data']['dev_path']
TEST_PATH = config['data']['test_path']

# Training
BATCH_SIZE = config['training']['batch_size']
NUM_EPOCHS = config['training']['num_epochs']
SAVE_DIR = config['training']['save_dir']

# Read STREUSLE data and create data sets
train_data = StreusleDataset(TRAIN_PATH)
dev_data = StreusleDataset(DEV_PATH)
test_data = StreusleDataset(TEST_PATH)

# Specify device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'

# Fetch the BIO-style labels that include MWE information and create
# a label dictionary that includes all labels (train, dev and test).
labels = [
    tok['lextag']
    for sent in train_data.sents + dev_data.sents + test_data.sents
    for tok in sent
]
unique_labels = sorted(list(set(labels)))
label_to_id = {l: i for i, l in enumerate(unique_labels)}
id_to_label = {i: l for l, i in label_to_id.items()}
id_to_label[-100] = '[IGNORE]'

# Instantiate BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Create data loaders for train and dev
data_loader_train = DataLoader(
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

data_loader_dev = DataLoader(
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
    model_name='bert-base-uncased',
    num_labels=len(label_to_id)
)


# Training loop
for epoch in trange(NUM_EPOCHS, desc='Epoch'):
    for step, batch in enumerate(tqdm(data_loader_train)):
        logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        print(logits.shape)
        exit()
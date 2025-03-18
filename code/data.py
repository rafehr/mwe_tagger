import json
from typing import Tuple, List, Dict
from pathlib import Path

from conllu import parse, TokenList
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

########################################################################
# Classes
########################################################################

class StreusleDataset(Dataset):
    def __init__(self, data_path: Path):
        self.sents = read_streusle_conllulex(data_path)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx: int) -> Dict[str, List[str]]:
        tokens = [tok['form'] for tok in self.sents[idx]]
        labels = [tok['lextag'] for tok in self.sents[idx]]
        
        return {'tokens': tokens, 'labels': labels}

########################################################################
# Functions
########################################################################

def collate_fn(
    batch: List[Dict[str, List[str]]],
    label_to_id: Dict[str, int],
    tokenizer: BertTokenizerFast,
    max_len: int
) -> BatchEncoding:
    """Takes in a batch of examples and tokenizes it batch-wise. Also,
    the labels are encoded and aligned with the tokenized input.
    """
    batch_tokens = [example['tokens'] for example in batch]
    batch_labels = [example['labels'] for example in batch]
    batch_enc_labels = [[label_to_id[l] for l in labels]
                        for labels in batch_labels]

    batch_encoding = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors='pt'
    )

    batch_aligned_labels = [
        align_labels(enc_labels, batch_encoding.word_ids(i))
        for i, enc_labels in enumerate(batch_enc_labels)
    ]
    batch_encoding['labels'] = torch.tensor(
        batch_aligned_labels, 
        dtype=torch.long
    )
    return batch_encoding 

def read_streusle_conllulex(file_path: Path) -> List[TokenList]:
    """Reads a STREUSLE conllulex file."""
    fields = (
        'id', 'form', 'lemma', 'upos',
        'xpos', 'feats', 'head', 'deprel',
        'deps', 'misc', 'smwe', 'lexcat',
        'lexlemma', 'ss', 'ss2', 'wmwe',
        'wcat', 'wlemma', 'lextag'
    )
    with open(file_path, 'r', encoding='utf-8') as f:
       sents = parse(f.read(), fields)
    return sents


def align_labels(
        enc_labels: List[int],
        word_ids: List[int | None]
) -> List[int]:
    """Aligns the labels with the tokenized input. Only the first of
    the subwords is labeled, the others receive the placeholder -100.

    Args:
        enc_labels: The original labels, already converted to integers
            but without padding.
            -------
            Example:
                [3, 4, 5]
        word_ids: A list that maps the tokenized input to the
            original token ids. Padding, [CLS] and [SEP] are None.
            -------
            Example:
                >>> encoding['input_ids'] # the tokenized input
                [101, 456, 9878, 3233, 79797, 0, 0]
                >>> encoding.word_ids()
                [None, 0, 1, 1, 2, None, None]

    Returns:
        aligned_labels: Returns the aligned labels (with padding).

    Example:
        >>> align_labels(labels, encoding.word_ids())
        [-100, 3, 4, -100, 5, -100, -100]
    """
    aligned_labels = []
    previous_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_id:
            aligned_labels.append(enc_labels[word_id])
        else:
            aligned_labels.append(-100)
        previous_id = word_id
    return aligned_labels


def ids_to_tokens(
        tokenizer: BertTokenizerFast,
        input_ids: List[int]
) -> str | List[str]:
    """Converts input ids back to strings.

    Example:
        >>> ids_to_tokens(
                self.tokenizer, encoding['input_ids'].squeeze(0)
            )
    """
    return tokenizer.convert_ids_to_tokens(input_ids)

def get_label_dict(
    data: List[TokenList]
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Creates dictionaries that map from labels to integers and
    from integers to labels. Also, it writes the label dictionaries
    to JSON files.
    """
    labels = [
        tok['lextag']
        for sent in data
        for tok in sent
    ]
    unique_labels = sorted(list(set(labels)))
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    id_to_label[-100] = '[IGNORE]'
    save_labels(label_to_id=label_to_id, id_to_label=id_to_label)
    return label_to_id, id_to_label

def save_labels(label_to_id: Dict[str, int], id_to_label: Dict[int, str]):
    """Writes the label dictionaries to a JSON file."""
    with open('label_to_id.json', 'w') as f:
        json.dump(label_to_id, f, indent=4)
    with open('id_to_label.json', 'w') as f:
        json.dump(id_to_label, f, indent=4)
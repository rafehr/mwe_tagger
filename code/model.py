import torch.nn as nn
from torch import Tensor
from transformers import BertModel

class MWETagger(nn.Module):
    def __init__(self,
        pretrained_model_name: str,
        num_labels: int,
        device: str
    ):
        super(MWETagger, self).__init__()
        self.num_labels = num_labels
        self.device = device
        self.base_model = BertModel.from_pretrained(pretrained_model_name)
        H = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(
            in_features=H,
            out_features=num_labels
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Returns the logits for a sequence of tokens.

        Args:
            input_ids: The token ids produced by the BERT tokenizer.
            attention_mask: The attention mask that tells the BERT
                model what to focus on.

        Returns:
            logits: The logits for a sequence of tokens. Every logit
                indicates whether a token belongs to a MWE or not.
        Returns:
            logits : The logits for a sequence of tokens. Every logit
                indicates whether a token belongs to a MWE or not.
        """
        # b x seq_len x hidden_size
        cont_reprs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0].to(self.device)
        cont_reprs = self.dropout(cont_reprs)
        # b x seq_len x num_labels
        logits = self.classifier(cont_reprs)
        return logits

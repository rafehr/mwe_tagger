import torch.nn as nn
from torch import Tensor
from transformers import BertModel

class MWETagger(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(MWETagger, self).__init__()
        self.num_labels = num_labels
        self.base_model = BertModel.from_pretrained(model_name)
        H = self.base_model.config.hidden_size
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
        )[0]
        # b x seq_len x num_labels
        logits = self.classifier(cont_reprs)
        return logits

import torch
from torch import nn


class RelationModel(nn.Module):
    def __init__(self, base_model: nn.Module, out_embedding_size: int, dropout_rate: float):
        super().__init__()

        self._base_model = base_model
        self._hidden_size = self._base_model.config.hidden_size

        self._dropout = nn.Dropout(dropout_rate)
        self._fclayer = nn.Linear(self._hidden_size * 3, out_embedding_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
    ):
        outputs = self._base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        token_type_ids = None
        if token_type_ids is None:
            relation_embeddings = torch.tanh(pooled_output)
            relation_embeddings = self._dropout(relation_embeddings)

            return relation_embeddings

        # Entities extraction
        # e1_h = self.extract_entity(sequence_output, e1_mask)
        # e2_h = self.extract_entity(sequence_output, e2_mask)

        # context = self._dropout(pooled_output)

        # pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        # pooled_output = torch.tanh(pooled_output)
        # pooled_output = self._fclayer(pooled_output)

        # relation_embeddings = torch.tanh(pooled_output)
        # relation_embeddings = self._dropout(relation_embeddings)

        # return relation_embeddings

    @staticmethod
    def extract_entity(sequence_output, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
        return extended_e_mask.float()

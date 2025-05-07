import torch
from torch import nn

from zeshrex.model.relation_model import RelationModel


class RelationClassifierModel(RelationModel):
    def __init__(self, base_model: nn.Module, num_classes: int, out_embedding_size: int, dropout_rate: float):
        super().__init__(base_model, out_embedding_size, dropout_rate)

        self._classifier = nn.Linear(out_embedding_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
    ):
        relation_embeddings = super().forward(input_ids, attention_mask, token_type_ids, e1_mask, e2_mask)
        logits = self._classifier(relation_embeddings)  # [batch_size x hidden_size]

        return logits, relation_embeddings

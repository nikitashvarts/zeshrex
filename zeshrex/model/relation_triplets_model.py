import torch
from torch import nn

from zeshrex.model.relation_model import RelationModel


class RelationTripletsModel(RelationModel):
    def __init__(self, base_model: nn.Module, out_embedding_size: int, dropout_rate: float):
        super().__init__(base_model, out_embedding_size, dropout_rate)

    def forward(
        self,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        anchor_token_type_ids: torch.Tensor,
        anchor_e1_mask: torch.Tensor,
        anchor_e2_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        pos_token_type_ids: torch.Tensor,
        pos_e1_mask: torch.Tensor,
        pos_e2_mask: torch.Tensor,
        neg_input_ids: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        neg_token_type_ids: torch.Tensor,
        neg_e1_mask: torch.Tensor,
        neg_e2_mask: torch.Tensor,
    ):
        anchor_embeddings = super().forward(
            anchor_input_ids,
            anchor_attention_mask,
            anchor_token_type_ids,
            anchor_e1_mask,
            anchor_e2_mask,
        )
        positive_embeddings = super().forward(
            pos_input_ids,
            pos_attention_mask,
            pos_token_type_ids,
            pos_e1_mask,
            pos_e2_mask,
        )
        negative_embeddings = super().forward(
            neg_input_ids,
            neg_attention_mask,
            neg_token_type_ids,
            neg_e1_mask,
            neg_e2_mask,
        )

        return anchor_embeddings, positive_embeddings, negative_embeddings

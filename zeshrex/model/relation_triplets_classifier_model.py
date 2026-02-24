import torch
from torch import nn

from zeshrex.model.relation_triplets_model import RelationTripletsModel


class RelationTripletsClassificationModel(RelationTripletsModel):
    def __init__(
        self,
        base_model: nn.Module,
        sentence_model: nn.Module,
        num_classes: int,
        out_embedding_size: int,
        dropout_rate: float,
    ):
        super().__init__(base_model, out_embedding_size, dropout_rate)

        self._sentence_model = sentence_model
        self._classifier = nn.Linear(out_embedding_size, num_classes)

        for param in self._sentence_model.parameters():
            param.requires_grad = False

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
        desc_input_ids: torch.Tensor,
        desc_attention_mask: torch.Tensor,
    ):
        anchor_embeddings, positive_embeddings, negative_embeddings = super().forward(
            anchor_input_ids,
            anchor_attention_mask,
            anchor_token_type_ids,
            anchor_e1_mask,
            anchor_e2_mask,
            pos_input_ids,
            pos_attention_mask,
            pos_token_type_ids,
            pos_e1_mask,
            pos_e2_mask,
            neg_input_ids,
            neg_attention_mask,
            neg_token_type_ids,
            neg_e1_mask,
            neg_e2_mask,
        )

        desc_embeddings = self._sentence_model(input_ids=desc_input_ids, attention_mask=desc_attention_mask)[
            1
        ]  # pooled output

        logits = self._classifier(anchor_embeddings)  # [batch_size x hidden_size]

        return anchor_embeddings, positive_embeddings, negative_embeddings, desc_embeddings, logits

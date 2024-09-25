from torch import nn
import torch


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

        # Entities extraction
        e1_h = self.extract_entity(sequence_output, e1_mask)
        e2_h = self.extract_entity(sequence_output, e2_mask)

        context = self._dropout(pooled_output)

        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self._fclayer(pooled_output)

        relation_embeddings = torch.tanh(pooled_output)
        relation_embeddings = self._dropout(relation_embeddings)

        return relation_embeddings

    @staticmethod
    def extract_entity(sequence_output, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
        return extended_e_mask.float()


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
            anchor_input_ids, anchor_attention_mask, anchor_token_type_ids, anchor_e1_mask, anchor_e2_mask
        )
        positive_embeddings = super().forward(
            pos_input_ids, pos_attention_mask, pos_token_type_ids, pos_e1_mask, pos_e2_mask,
        )
        negative_embeddings = super().forward(
            neg_input_ids, neg_attention_mask, neg_token_type_ids, neg_e1_mask, neg_e2_mask,
        )

        return anchor_embeddings, positive_embeddings, negative_embeddings

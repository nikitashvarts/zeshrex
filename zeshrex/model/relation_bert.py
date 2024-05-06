import torch
from torch import nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

from zeshrex.layers import FCLayer


class RelationBert(nn.Module):
    def __init__(
        self,
        config: BertConfig,
        output_size: int,
        dropout_rate: float,
        alpha: float = 0.4,
        gamma: float = 7.5
    ):
        super().__init__()
        self._bert_sample = BertModel(config=config)  # Load pretrained bert
        self._bert_relation = BertModel(config=config)

        self.num_labels = config.num_labels
        self._alpha = alpha
        self._gamma = gamma

        self.cls_fc_layer = FCLayer(
            input_dim=config.hidden_size,
            output_dim=config.hidden_size,
            dropout_rate=dropout_rate,
        )
        self.entity_fc_layer = FCLayer(
            input_dim=config.hidden_size,
            output_dim=config.hidden_size,
            dropout_rate=dropout_rate,
        )
        self.dim_reduction_fc_layer = FCLayer(
            input_dim=config.hidden_size * 3,
            output_dim=output_size,
            dropout_rate=dropout_rate,
        )
        self.label_classifier = FCLayer(
            input_dim=output_size,
            output_dim=config.num_labels,
            hidden_dim=output_size // 2,
            dropout_rate=dropout_rate,
            use_activation=True,
        )

    def forward(
            self,
            # input_ids, attention_masks, token_type_ids, e1_masks, e2_masks, labels
            anchor_input_ids, anchor_attention_masks, anchor_token_type_ids, anchor_e1_masks, anchor_e2_masks,
            pos_input_ids, pos_attention_masks, pos_token_type_ids, pos_e1_masks, pos_e2_masks,
            neg_input_ids, neg_attention_masks, neg_token_type_ids, neg_e1_masks, neg_e2_masks,
            desc_input_ids, desc_attention_masks,
            labels=None,
    ):
        # TODO: make options, classification or triplet loss

        # -------------------------------------------------------------------
        anchor_embeddings = self._get_relation_embedding(
            anchor_input_ids, anchor_attention_masks, anchor_token_type_ids, anchor_e1_masks, anchor_e2_masks
        )
        positive_embeddings = self._get_relation_embedding(
            pos_input_ids, pos_attention_masks, pos_token_type_ids, pos_e1_masks, pos_e2_masks,
        )
        negative_embeddings = self._get_relation_embedding(
            neg_input_ids, neg_attention_masks, neg_token_type_ids, neg_e1_masks, neg_e2_masks,
        )

        # relation_desc_embedding = self._bert_relation(desc_input_ids, desc_attention_masks)[1]

        distance_pos = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings)
        distance_neg = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings)

        triplet_loss = torch.nn.functional.relu(distance_pos - distance_neg + self._gamma).mean()

        # desc_loss = torch.nn.functional.pairwise_distance(anchor_embeddings, relation_desc_embedding).mean()

        # -------------------------------------------------------------------
        if labels is not None:
            logits = self.label_classifier(anchor_embeddings)
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                softmax_loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                softmax_loss = loss_fct(logits, labels)
        else:
            softmax_loss = 0
        # -------------------------------------------------------------------

        # loss = (1 - self._alpha) * (triplet_loss + desc_loss) + self._alpha * softmax_loss  # TODO: plus or minus softmax?
        loss = (1 - self._alpha) * (triplet_loss) + self._alpha * softmax_loss  # TODO: plus or minus softmax?

        return loss, anchor_embeddings

        # reduced_concat_h = self._get_relation_embedding(
        #     input_ids, attention_masks, token_type_ids, e1_masks, e2_masks
        # )
        # logits = self.label_classifier(reduced_concat_h)
        #
        # outputs = (logits,)  # + outputs[2:]  # add hidden states and attention if they are here  # TODO: is it needed?
        #
        # # Softmax
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(logits, labels)
        #
        #     outputs = (loss,) + outputs
        #
        # return loss, reduced_concat_h

    def _get_relation_embedding(self, input_ids, attention_masks, token_type_ids, e1_masks, e2_masks):
        outputs = self._bert_sample(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # if token_type_ids is None and e1_masks is None and e2_masks is None:
        #     pooled_output = self.cls_fc_layer(pooled_output)
        #     return pooled_output

        # Average
        e1_h = self._entity_average(sequence_output, e1_masks)
        e2_h = self._entity_average(sequence_output, e2_masks)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer -> classifier_fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        reduced_concat_h = self.dim_reduction_fc_layer(concat_h)

        return reduced_concat_h
        # return pooled_output

    @staticmethod
    def _entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

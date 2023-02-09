import torch
from torch import nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

from zeshrex.layers import FCLayer


class RelationBert(BertPreTrainedModel):
    def __init__(self, config: BertConfig, output_size: int, dropout_rate: float):
        super().__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        self.dim_reduction_fc_layer = FCLayer(config.hidden_size * 3, output_size, dropout_rate)
        self.label_classifier = FCLayer(
            output_size,
            config.num_labels,
            dropout_rate,
            use_activation=False,
        )

    def forward(self, input_ids, attention_mask, token_type_ids, e1_mask, e2_mask, labels):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self._entity_average(sequence_output, e1_mask)
        e2_h = self._entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer -> classifier_fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        reduced_concat_h = self.dim_reduction_fc_layer(concat_h)
        logits = self.label_classifier(reduced_concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs, reduced_concat_h  # (loss), logits, (hidden_states), (attentions)

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

import logging
import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig

from zeshrex import PROJECT_PATH
from zeshrex.data.datasets import RelationDataset, collate_data
from zeshrex.data.preprocessing import RelationTokenizationPreprocessor
from zeshrex.layers.fc_layer import FCLayer
from zeshrex.utils.logger import init_logger


# Define the model
class RelationClassifier(nn.Module):
    def __init__(self, model_name, hidden_size, num_classes):
        super(RelationClassifier, self).__init__()

        self.bert_config = BertConfig.from_pretrained(model_name, num_labels=num_classes)
        self.bert = BertModel.from_pretrained(model_name, config=self.bert_config)

        self.dropout = nn.Dropout(dropout_rate)
        self.fclayer = nn.Linear(self.bert_config.hidden_size * 3, relation_embedding_dim)
        self.classifier = nn.Linear(relation_embedding_dim, num_classes)

        # -------------------------------------------------------------
        # self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, num_classes)

        # self.cls_fc_layer = FCLayer(
        #     input_dim=hidden_size,
        #     output_dim=hidden_size,
        #     dropout_rate=dropout_rate,
        # )
        # self.entity_fc_layer = FCLayer(
        #     input_dim=hidden_size,
        #     output_dim=hidden_size,
        #     dropout_rate=dropout_rate,
        # )
        # self.dim_reduction_fc_layer = FCLayer(
        #     input_dim=hidden_size * 3,
        #     output_dim=output_size,
        #     dropout_rate=dropout_rate,
        # )

    def forward(self, input_ids, attention_masks, token_type_ids, e1_masks, e2_masks, labels):
        # Pass the inputs through BERT
        outputs = self.bert(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)

        # Sequence of hidden-states of the last layer.
        sequence_output = outputs[0]

        # Last layer hidden-state of the [CLS] token further processed by a Linear layer and a Tanh activation function.
        pooled_output = outputs[1]

        # Entities extraction
        e1_h = self.extract_entity(sequence_output, e1_masks)
        e2_h = self.extract_entity(sequence_output, e2_masks)

        context = self.dropout(pooled_output)

        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.fclayer(pooled_output)

        relation_embeddings = torch.tanh(pooled_output)
        relation_embeddings = self.dropout(relation_embeddings)

        logits = self.classifier(relation_embeddings)  # [batch_size x hidden_size]

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs, relation_embeddings  # (loss), logits, (hidden_states), (attentions)

        # -------------------------------------------------------------
        # # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        # pooled_output = self.cls_fc_layer(pooled_output)
        # e1_h = self.entity_fc_layer(e1_h)
        # e2_h = self.entity_fc_layer(e2_h)

        # # Concat -> fc_layer -> classifier_fc_layer
        # concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        # reduced_concat_h = self.dim_reduction_fc_layer(concat_h)

        # # Use the [CLS] embedding as the sentence representation
        # out = reduced_concat_h  # [:, 0, :]
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # return out

    @staticmethod
    def extract_entity(sequence_output, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(extended_e_mask.float(), sequence_output).squeeze(1)
        return extended_e_mask.float()


def eval_model(model, device, test_dataloader, criterion):
    logging.info('==========')
    logging.info('Evaluation')
    logging.info('==========')

    model.eval()
    
    total_loss, total_accuracy = 0, 0
    total_preds = []
    total_labels = []

    for step, batch in enumerate(test_dataloader):
        if step % 10 == 0 and not step == 0:
            logging.info('Evaluation: Batch {:>5,} of {:>5,}'.format(step, len(test_dataloader)))

        batch = tuple(t.to(device) for t in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_masks': batch[1],
            'token_type_ids': batch[2],
            'e1_masks': batch[3],
            'e2_masks': batch[4],
            'labels': batch[5],
        }
        labels = batch[5]

        with torch.no_grad():
            outputs, _ = model(**inputs)

        loss = criterion(outputs[0], labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs[0], dim=1).flatten()
        total_preds.extend(preds.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

    print(set(total_preds), set(total_labels))

    avg_loss = total_loss / len(test_dataloader)
    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')

    return avg_loss, precision, recall, f1


def run_baseline():
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    text_preprocessor = RelationTokenizationPreprocessor(
        tokenizer=tokenizer, max_len=max_len, relation_tokens=['<e1>', '</e1>', '<e2>', '</e2>']
    )

    dataset = RelationDataset.from_directory(dir_path=PROJECT_PATH / dataset_path, text_processor=text_preprocessor)

    train_dataset, test_dataset, val_dataset = dataset.generate_train_test_split(
        use_predefined_split=use_predefined_split, use_zero_shot_split=use_zero_shot_split
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_data,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_data,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function and optimizer
    model = RelationClassifier(model_name=model_name, hidden_size=hidden_size, num_classes=len(train_dataset.labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.to(device)

    global_steps_count = 0
    steps_per_epoch = len(train_loader)

    losses = []

    # Training loop
    # -------------
    for epoch in range(num_epochs):
        logging.info('========')
        logging.info(f'EPOCH {epoch + 1}')
        logging.info('========')

        running_loss = 0.0
        steps_count = 0
        for batch in train_loader:
            global_steps_count += 1
            steps_count += 1

            model.train()

            batch = tuple(t.to(device) for t in batch)

            inputs = {
                'input_ids': batch[0],
                'attention_masks': batch[1],
                'token_type_ids': batch[2],
                'e1_masks': batch[3],
                'e2_masks': batch[4],
                'labels': batch[5],
            }
            labels = batch[5]

            outputs, relation_embedding = model(**inputs)
            loss = criterion(outputs[0], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            avg_loss = running_loss / steps_count
            losses.append(avg_loss)
            logging.info(
                'Epoch {:^3} Step {:^5} --- '
                'Average loss (over {:^5} training steps out of {}): {:.5f}'.format(
                    epoch + 1,
                    global_steps_count,
                    steps_count,
                    steps_per_epoch,
                    running_loss / steps_count,
                )
            )
            # Plot the loss
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.ylabel('Loss')
            plt.xlabel('Step')
            plt.title('Training Loss')
            (PROJECT_PATH / 'output').mkdir(exist_ok=True, parents=True)
            plt.savefig(PROJECT_PATH / 'output' / 'loss_plot_webnlg.png')
            # plt.savefig(PROJECT_PATH / 'output' / 'loss_plot_nyt.png')
            plt.close()

            # if steps_count % len(train_loader) == 0:
            if steps_count % 10 == 0:
                # Calculate metrics on the validation set
                avg_loss, precision, recall, f1 = eval_model(model, device, test_loader, criterion)
                logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
                logging.info(f'Validation Loss: {avg_loss}, Precision: {precision}, Recall: {recall}, F1-score: {f1}')


if __name__ == "__main__":
    init_logger(file_name='run_pipeline.log', level=logging.INFO)

    gpu = 0

    model_name = 'bert-large-cased'
    max_len = 200
    dataset_path = './datasets/prepared/WebNLG/'
    # dataset_path = './datasets/prepared/NYT/'
    use_predefined_split = True
    use_zero_shot_split = False

    relation_embedding_dim = 1024

    batch_size = 32
    eval_batch_size = 32
    num_epochs = 10

    hidden_size = 768
    output_size = 768
    dropout_rate = 0.1  # 0.25

    model_config = {
        'name': 'bert-base-cased',
        'use_pretrain': '',
        'output_size': 512,
        'dropout_rate': 0.25,
        'hidden_size': 512,
        'hidden_act': 'gelu',
        'initializer_range': 0.02,
        'vocab_size': 30522,
        'hidden_dropout_prob': 0.1,
        'num_attention_heads': 8,
        'type_vocab_size': 2,
        'max_position_embeddings': 512,
        'num_hidden_layers': 4,
        'intermediate_size': 2048,
        'attention_probs_dropout_prob': 0.0,
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    run_baseline()

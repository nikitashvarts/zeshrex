import logging
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig

from zeshrex import PROJECT_PATH
from zeshrex.data.datasets import RelationDataset, collate_data
from zeshrex.data.preprocessing import RelationTokenizationPreprocessor
from zeshrex.utils import init_logger


class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels):
        super(TextClassifier, self).__init__()

        self.bert = BertModel(config=BertConfig())  # Load pretrained bert

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_masks, token_type_ids, *args, **kwargs):
        outputs = self.bert(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        out = self.fc1(pooled_output)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        return out


def main():
    # Define hyperparameters
    input_size = 768  # Size of BERT embeddings
    hidden_size = 256
    learning_rate = 0.0002
    num_epochs = 10
    batch_size = 4
    max_len = 200

    model_name = 'bert-large-cased'
    dataset_path = PROJECT_PATH / './datasets/prepared/WebNLG/'

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    text_preprocessor = RelationTokenizationPreprocessor(
        tokenizer=tokenizer, max_len=max_len, relation_tokens=['<e1>', '</e1>', '<e2>', '</e2>']
    )

    dataset = RelationDataset.from_directory(
        dir_path=PROJECT_PATH / dataset_path, text_processor=text_preprocessor
    )

    train_dataset, test_dataset, val_dataset = dataset.generate_train_test_split(use_predefined_split=True)

    train_loader = DataLoader(
        dataset=train_dataset,  # train_triplets_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_data,
    )
    test_loader = DataLoader(
        dataset=test_dataset,  # test_triplets_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_data,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an instance of the classifier
    model = TextClassifier(input_size, hidden_size, num_labels=len(dataset.labels))
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    global_steps_count = 0
    steps_per_epoch = len(train_loader)
    for epoch in range(num_epochs):
        logging.info('========')
        logging.info(f'EPOCH {epoch + 1}')
        logging.info('========')
        model.train()
        running_loss = 0.0
        steps_count = 0
        for i, data in enumerate(train_loader):
            global_steps_count += 1
            steps_count += 1
            batch = tuple(t.to(device) for t in data)

            inputs = {
                'input_ids': batch[0],
                'attention_masks': batch[1],
                'token_type_ids': batch[2],
            }
            targets = batch[5]

            # Forward pass
            outputs = model(**inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps_count % 50 == 0:
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

        # Print loss after every epoch
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # After training, you can use the trained classifier to make predictions
        val_result = eval_model(model, device, test_loader, relations=dataset.relations_encoding)
        logging.info(val_result)


def eval_model(
        model: BertModel, device: torch.device, dataloader: DataLoader, relations: Dict[str, int]
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        metrics = {}
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_masks': batch[1],
                'token_type_ids': batch[2],
                'e1_masks': batch[3],
                'e2_masks': batch[4],
                'labels': batch[5],
            }

            outputs = model(**inputs)
            probabilities = nn.functional.softmax(outputs, dim=-1)

            pred_labels = probabilities.cpu().detach().numpy().argmax(axis=1)
            true_labels = inputs['labels'].cpu().detach().numpy()

            precision, recall, f1_score, support = precision_recall_fscore_support(
                y_true=true_labels,
                y_pred=pred_labels,
                labels=list(relations.values()),
                zero_division=0,
            )

            metrics['precision'] = metrics.get('precision', []) + [precision]
            metrics['recall'] = metrics.get('recall', []) + [recall]
            metrics['f1_score'] = metrics.get('f1_score', []) + [f1_score]
            # metrics['loss'] = metrics.get('loss', []) + [float(loss)]

        precision_batch_mean = np.mean(metrics['precision'], axis=0)
        recall_batch_mean = np.mean(metrics['recall'], axis=0)
        f1_score_batch_mean = np.mean(metrics['f1_score'], axis=0)

        logging.info('| {:^30} | {:^11} | {:^10} | {:^10} |'.format('Relation', 'Precision', 'Recall', 'F1-score'))
        for relation, p, r, f1 in zip(relations.keys(), precision_batch_mean, recall_batch_mean, f1_score_batch_mean):
            logging.info('| {:^30} | {:^11.5f} | {:^10.5f} | {:^10.5f} |'.format(relation, p, r, f1))

        sorted_index = np.argsort(f1_score_batch_mean)[::-1][:5]
        logging.info(
            'TOP-5 macro average: '
            f'Precision: {np.mean(precision_batch_mean[sorted_index])} | '
            f'Recall: {np.mean(recall_batch_mean[sorted_index])} | '
            f'F1-score: {np.mean(f1_score_batch_mean[sorted_index])}'
        )

        precision_macro = np.mean(precision_batch_mean)
        recall_macro = np.mean(recall_batch_mean)
        f1_score_macro = np.mean(f1_score_batch_mean)
        # avg_loss = np.mean(metrics['loss'])

        results = {
            # 'eval_loss': np.round(avg_loss, 5),
            'f1_score_macro': np.round(f1_score_macro, 5),
            'precision_macro': np.round(precision_macro, 5),
            'recall_macro': np.round(recall_macro, 5),
        }
    return results


if __name__ == '__main__':
    init_logger(level=logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    main()

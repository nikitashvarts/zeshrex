import logging
from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from zeshrex.model import Model


def eval_model(model: Model, device: torch.device, dataloader: DataLoader, relations: Dict[str, int]) -> Dict[str, Any]:
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
                'labels': batch[5]
            }

            (loss, logits), embeddings = model(**inputs)
            probabilities = nn.functional.softmax(logits, dim=-1)

            pred_labels = probabilities.cpu().detach().numpy().argmax(axis=1)
            true_labels = inputs['labels'].cpu().detach().numpy()

            precision, recall, f1_score, support = precision_recall_fscore_support(
                y_true=true_labels, y_pred=pred_labels, labels=list(relations.values()), zero_division=0,
            )

            metrics['precision'] = metrics.get('precision', []) + [precision]
            metrics['recall'] = metrics.get('recall', []) + [recall]
            metrics['f1_score'] = metrics.get('f1_score', []) + [f1_score]
            metrics['loss'] = metrics.get('loss', []) + [float(loss)]

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
        avg_loss = np.mean(metrics['loss'])

        results = {
            'eval_loss': np.round(avg_loss, 5),
            'f1_score_macro': np.round(f1_score_macro, 5),
            'precision_macro': np.round(precision_macro, 5),
            'recall_macro': np.round(recall_macro, 5),
        }
    return results

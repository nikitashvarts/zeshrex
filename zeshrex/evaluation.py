from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from zeshrex.model import Model


def eval_model(model: Model, device: torch.device, dataloader: DataLoader) -> Dict[str, Any]:
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
                y_true=true_labels, y_pred=pred_labels, labels=dataloader.dataset.labels, zero_division=0,
            )

            metrics['precision'] = metrics.get('precision', []) + [precision]
            metrics['recall'] = metrics.get('recall', []) + [recall]
            metrics['f1_score'] = metrics.get('f1_score', []) + [f1_score]
            metrics['loss'] = metrics.get('loss', []) + [float(loss)]

        precision_macro = np.mean(np.mean(metrics['precision'], axis=0))
        recall_macro = np.mean(np.mean(metrics['recall'], axis=0))
        f1_score_macro = np.mean(np.mean(metrics['f1_score'], axis=0))
        avg_loss = np.mean(metrics['loss'])

        results = {
            'eval_loss': np.round(avg_loss, 5),
            'f1_score_macro': np.round(f1_score_macro, 5),
            'precision_macro': np.round(precision_macro, 5),
            'recall_macro': np.round(recall_macro, 5),
        }
    return results

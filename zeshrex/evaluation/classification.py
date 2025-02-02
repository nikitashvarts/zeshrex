import logging
import os
from types import SimpleNamespace
from typing import List, Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from zeshrex.evaluation.common import calculate_classification_metrics, measure_clusters_distances, visualize_clusters
from zeshrex.model import Model


def eval_classification_model(
    cfg: SimpleNamespace,
    model: Model,
    device: torch.device,
    dataloader: DataLoader,
    relations: Dict[str, int],
    criterion: nn.Module,
    output_dir: Optional[os.PathLike] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    logging.info('==========')
    logging.info('Evaluation')
    logging.info('==========')

    model.eval()
    label_to_relation: Dict[int, str] = {v: k for k, v in relations.items()}

    softmax = torch.nn.Softmax(dim=1)

    steps_per_evaluation = len(dataloader)
    running_loss: float = 0.0
    steps_count: int = 0

    total_preds = []
    total_labels = []
    
    metrics = {}
    embeddings_clusters: Dict[int, List[np.ndarray]] = {}

    for batch in dataloader:
        steps_count += 1

        batch = tuple(t.to(device) for t in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'e1_mask': batch[3],
            'e2_mask': batch[4],
        }
        labels = batch[5]

        with torch.no_grad():
            logits, relation_embeddings = model(**inputs)

            loss = criterion(logits, labels)
            running_loss += loss.item()

            probs = softmax(logits)
            preds = torch.argmax(probs, dim=1)

            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

            embeddings_batch_arr = relation_embeddings.cpu().detach().numpy()
            labels_arr = labels.cpu().numpy()
            # note for line above: we want to see how true labels are distributed among classified clusters

        for label, embedding in zip(labels_arr, embeddings_batch_arr):
            embeddings_clusters[label] = embeddings_clusters.get(label, []) + [embedding]

        metrics['loss'] = metrics.get('loss', []) + [float(loss)]

        if steps_count % cfg.general.log_frequency == 0:
            logging.info(
                'Evaluation step {:^5} out of {} --- '
                'Average loss: {:.5f}'.format(
                    steps_count,
                    steps_per_evaluation,
                    running_loss / steps_count,
                )
            )

    avg_loss = np.mean(metrics['loss'])

    precision_macro, recall_macro, f1_score_macro = calculate_classification_metrics(
        total_labels, total_preds, relations
    )

    avg_inner_distance, avg_outer_distance = measure_clusters_distances(embeddings_clusters, relations)
    visualize_clusters(embeddings_clusters, label_to_relation, output_dir, tag)

    results = {
        'eval_loss': np.round(avg_loss, 5),
        'f1_score_macro': np.round(f1_score_macro, 5),
        'precision_macro': np.round(precision_macro, 5),
        'recall_macro': np.round(recall_macro, 5),
        'avg_inner_distance': np.round(avg_inner_distance, 7),
        'avg_outer_distance': np.round(avg_outer_distance, 7),
    }

    return results

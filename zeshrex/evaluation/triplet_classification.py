import logging
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from zeshrex.evaluation.common import (
    calculate_classification_metrics,
    measure_clusters_distances,
    visualize_clusters,
)
from zeshrex.model import Model
from zeshrex.training.tools import select_hard_negatives


def calculate_text_embeddings(
    sentence_model: nn.Module,
    label_to_description_tokens: Dict[int, Tuple[List[int], List[int]]],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Calculate text embeddings for relation descriptions using the sentence model.

    Args:
        sentence_model (nn.Module): The model used to calculate embeddings.
        relation_descriptions (Dict[str, str]): A dictionary of relation descriptions.
        device (torch.device): The device to run the model on.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of relation descriptions and their corresponding embeddings.
    """
    sentence_model.eval()
    embeddings: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for label, description_tokens in label_to_description_tokens.items():
            # For simplicity make batch size of 1
            input_ids = torch.unsqueeze(torch.tensor(description_tokens[0], dtype=torch.long), 0).to(device)
            attention_mask = torch.unsqueeze(torch.tensor(description_tokens[1], dtype=torch.long), 0).to(device)

            inputs_description = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            outputs = sentence_model(**inputs_description)
            embeddings[label] = outputs[1].squeeze()  # pooled output

    return embeddings


def find_closest_relation_labels(anchor_embeddings, label_to_relation_embedding):
    # Stack all relation embeddings into a single tensor
    labels = list(label_to_relation_embedding.keys())
    relation_embeddings = torch.stack(list(label_to_relation_embedding.values()))

    # Normalize the relation embeddings
    relation_embeddings = torch.nn.functional.normalize(relation_embeddings, p=2, dim=1)

    closest_labels = []
    for anchor_embedding in anchor_embeddings:
        # Normalize the anchor embedding
        anchor_embedding = torch.nn.functional.normalize(anchor_embedding, p=2, dim=0)

        # Compute cosine similarity using matrix multiplication
        similarities = torch.mm(anchor_embedding.unsqueeze(0), relation_embeddings.t()).squeeze(0)

        # Find the label with the highest similarity
        max_similarity, max_index = torch.max(similarities, dim=0)
        closest_label = labels[max_index.item()]
        closest_labels.append(closest_label)

    return closest_labels


def eval_metric_classification_model(
    cfg: SimpleNamespace,
    model: Model,
    sentence_model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    relation_labels: Dict[str, int],
    relation_descriptions_tokens: Dict[str, Tuple[List[int], List[int]]],
    criterion: nn.Module,
    output_dir: Optional[os.PathLike] = None,
    tag: Optional[str] = None,
):
    logging.info('==========')
    logging.info('Evaluation')
    logging.info('==========')

    model.eval()
    label_to_relation: Dict[int, str] = {v: k for k, v in relation_labels.items()}
    label_to_description_tokens: Dict[int, Tuple[List[int], List[int]]] = {
        relation_labels[relation]: tokens for relation, tokens in relation_descriptions_tokens.items()
    }
    label_to_relation_embedding: Dict[int, torch.Tensor] = calculate_text_embeddings(
        sentence_model, label_to_description_tokens, device
    )

    steps_per_evaluation = len(dataloader)
    running_loss: float = 0.0
    steps_count: int = 0

    total_preds = []
    total_labels = []

    metrics = {}
    embeddings_clusters: Dict[int, List[np.ndarray]] = {}

    for _, batch in enumerate(dataloader):
        steps_count += 1

        batch = tuple(t.to(device) for t in batch)

        inputs_relation = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'e1_mask': batch[3],
            'e2_mask': batch[4],
        }
        labels = batch[5]
        inputs_description = {
            'input_ids': batch[6],
            'attention_mask': batch[7],
        }

        with torch.no_grad():
            logits, anchor_embeddings = model(**inputs_relation)  # TODO: make logits options for Zero-Shot
            desc_embeddings = sentence_model(**inputs_description)[1]  # pooled output

            negative_embeddings = select_hard_negatives(anchor_embeddings, labels, device, margin=0.5, top_k=1)

            loss = criterion(anchor_embeddings, desc_embeddings, negative_embeddings, logits, labels)

            running_loss += loss.item()

            labels_arr = labels.cpu().detach().numpy()
            embeddings_batch_arr = anchor_embeddings.cpu().detach().numpy()

            total_preds.extend(find_closest_relation_labels(anchor_embeddings, label_to_relation_embedding))
            total_labels.extend(labels.cpu().numpy())

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
        total_labels, total_preds, relation_labels
    )

    avg_inner_distance, avg_outer_distance = measure_clusters_distances(embeddings_clusters, relation_labels)
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

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from zeshrex.model import Model


def viz_clusters(dataset, labels, cluster_centers):
    """Visualizes clusters, their centers, and distances using t-SNE.

    Args:
        dataset (np.array): A numpy array of input data.
        labels (np.array): A numpy array of cluster labels.
        cluster_centers (np.array): A numpy array of cluster centers.
    """

    # Project the input data and cluster centers to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    projected_data = tsne.fit_transform(dataset)
    projected_centers = tsne.transform(cluster_centers)

    # Plot the clusters, their centers, and the distances
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(set(labels))):
        mask = (labels == i)
        ax.scatter(projected_data[mask, 0], projected_data[mask, 1], c=f'{i}', s=10)
        ax.scatter(projected_centers[i, 0], projected_centers[i, 1], c=f'{i}', s=100, marker='*', edgecolor='black')

        avg_distance = np.mean(np.linalg.norm(projected_data[mask] - projected_centers[i], axis=1))
        ax.text(
            projected_centers[i, 0],
            projected_centers[i, 1] + 2,
            f'Avg. distance: {avg_distance:.2f}', ha='center',
            fontsize=12
        )

        min_distance = np.min(
            np.linalg.norm(projected_centers[np.arange(len(cluster_centers)) != i] - projected_centers[i], axis=1)
        )
        ax.text(projected_centers[i, 0], projected_centers[i, 1] - 8, f'Min. distance: {min_distance:.2f}', ha='center',
                fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def visualize_clusters(
        label_to_cluster: Dict[int, List[np.ndarray]],
        label_to_relation: Dict[int, str],
        output_dir: Optional[os.PathLike],
        tag: Optional[str] = None,
):
    embeddings = []
    labels = []

    for label, embeddings_list in label_to_cluster.items():
        embeddings.extend(embeddings_list)
        labels.extend([label] * len(embeddings_list))

    embeddings_arr = np.array(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    projected_data = tsne.fit_transform(embeddings_arr)

    grouped = {}
    for proj, label in zip(projected_data, labels):
        grouped[label] = grouped.get(label, []) + [proj]

    fig, ax = plt.subplots(figsize=(10, 10))
    for label, proj_list in grouped.items():
        proj = np.array(proj_list)
        ax.scatter(proj[:, 0], proj[:, 1], c=f'C{label}', s=10, label=label_to_relation[label])

    plt.legend()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / 'eval_clusters.png'
        if tag is not None:
            file_path = file_path.parent / f'{file_path.stem}_{tag}{file_path.suffix}'
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()


def eval_zero_shot_model(
        model: Model,
        device: torch.device,
        dataloader: DataLoader,
        relations: Dict[str, int],
        output_dir: Optional[os.PathLike] = None,
        tag: Optional[str] = None,
):
    model.eval()
    label_to_relation: [int, str] = {v: k for k, v in relations.items()}
    with torch.no_grad():
        metrics = {}
        embeddings_clusters: Dict[int, List[np.ndarray]] = {}
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)

            # inputs = {
            #     'input_ids': batch[0],
            #     'attention_masks': batch[1],
            #     'token_type_ids': batch[2],
            #     'e1_masks': batch[3],
            #     'e2_masks': batch[4],
            #     'labels': batch[5],
            # }
            # labels = batch[5]

            inputs = {
                'anchor_input_ids': batch[0],
                'anchor_attention_masks': batch[1],
                'anchor_token_type_ids': batch[2],
                'anchor_e1_masks': batch[3],
                'anchor_e2_masks': batch[4],
                'pos_input_ids': batch[5],
                'pos_attention_masks': batch[6],
                'pos_token_type_ids': batch[7],
                'pos_e1_masks': batch[8],
                'pos_e2_masks': batch[9],
                'neg_input_ids': batch[10],
                'neg_attention_masks': batch[11],
                'neg_token_type_ids': batch[12],
                'neg_e1_masks': batch[13],
                'neg_e2_masks': batch[14],
                'labels': batch[15],
                'desc_input_ids': batch[16],
                'desc_attention_masks': batch[17],
            }
            labels = batch[15]

            loss, embeddings_batch = model(**inputs)

            # labels_arr = inputs['labels'].cpu().detach().numpy()
            labels_arr = labels.cpu().detach().numpy()

            embeddings_batch_arr = embeddings_batch.cpu().detach().numpy()
            for label, embedding in zip(labels_arr, embeddings_batch_arr):
                embeddings_clusters[label] = embeddings_clusters.get(label, []) + [embedding]

            metrics['loss'] = metrics.get('loss', []) + [float(loss)]

        cluster_centers: Dict[int, np.ndarray] = {}
        for label, embeddings_list in embeddings_clusters.items():
            cluster_centers[label] = np.mean(embeddings_list, axis=0)

        distances = {}
        for label, data in embeddings_clusters.items():
            center = cluster_centers[label]
            distance_to_center = np.linalg.norm(data - center, axis=1)
            distances[label] = {
                "inner_distance": np.mean(distance_to_center),
                "outer_distance": np.min(
                    [
                        np.linalg.norm(center - other_center)
                        for other_label, other_center in cluster_centers.items()
                        if other_label != label
                    ]
                )
            }

        index_to_relation = {v: k for k, v in relations.items()}
        logging.info('| {:^30} | {:^20} | {:^20} |'.format('Relation', 'Inner Dist', 'Min Outer Dist'))
        for relation_index, dist_data in distances.items():
            inner = dist_data['inner_distance']
            outer = dist_data['outer_distance']
            logging.info(
                '| {:^30} | {:^20.7f} | {:^20.7f} |'.format(index_to_relation[relation_index], inner, outer)
            )

        avg_loss = np.mean(metrics['loss'])
        results = {
            'eval_loss': np.round(avg_loss, 5),
            'avg_inner_distance': np.round(np.mean([data['inner_distance'] for _, data in distances.items()]), 7),
            'avg_outer_distance': np.round(np.mean([data['outer_distance'] for _, data in distances.items()]), 7),
        }

        visualize_clusters(embeddings_clusters, label_to_relation, output_dir, tag)

        return results


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
                'labels': batch[5],
            }

            (loss, logits), embeddings = model(**inputs)
            probabilities = nn.functional.softmax(logits, dim=-1)

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

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support


def calculate_classification_metrics(labels, preds, relations):
    logging.info('')
    logging.info('Classification Metrics')
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true=labels,
        y_pred=preds,
        labels=list(relations.values()),
        zero_division=0,
    )

    logging.info('--------------------------------------------------------------------------')
    logging.info('| {:^30} | {:^11} | {:^10} | {:^10} |'.format('Relation', 'Precision', 'Recall', 'F1-score'))
    logging.info('--------------------------------------------------------------------------')
    for relation, p, r, f1 in zip(relations.keys(), precision, recall, f1_score):
        logging.info('| {:^30} | {:^11.5f} | {:^10.5f} | {:^10.5f} |'.format(relation, p, r, f1))
    logging.info('--------------------------------------------------------------------------')

    sorted_index = np.argsort(f1_score)[::-1][:5]
    logging.info(
        'TOP-5 macro average: '
        f'Precision: {np.mean(precision[sorted_index])} | '
        f'Recall: {np.mean(recall[sorted_index])} | '
        f'F1-score: {np.mean(f1_score[sorted_index])}'
    )

    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_score_macro = np.mean(f1_score)

    return precision_macro, recall_macro, f1_score_macro


def measure_clusters_distances(embeddings_clusters: Dict[int, List[np.ndarray]], relations: Dict[str, int]):
    logging.info('')
    logging.info('Clusters Distances')
    
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
            ),
        }

    index_to_relation = {v: k for k, v in relations.items()}
    logging.info('--------------------------------------------------------------------------------')
    logging.info('| {:^30} | {:^20} | {:^20} |'.format('Relation', 'Inner Dist', 'Min Outer Dist'))
    logging.info('--------------------------------------------------------------------------------')
    for relation_index, dist_data in distances.items():
        inner = dist_data['inner_distance']
        outer = dist_data['outer_distance']
        logging.info('| {:^30} | {:^20.7f} | {:^20.7f} |'.format(index_to_relation[relation_index], inner, outer))
    logging.info('--------------------------------------------------------------------------------')

    avg_inner_distance = np.mean([data['inner_distance'] for _, data in distances.items()])
    avg_outer_distance = np.mean([data['outer_distance'] for _, data in distances.items()])

    return avg_inner_distance, avg_outer_distance


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
        ax.scatter(proj[:, 0], proj[:, 1], c=f'C{label}', s=70, label=label_to_relation[label])

    plt.legend()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / 'eval_clusters.png'
        if tag is not None:
            file_path = file_path.parent / f'{file_path.stem}_{tag}{file_path.suffix}'
        plt.savefig(file_path, bbox_inches='tight')

    plt.show()


def viz_clusters_old(dataset, labels, cluster_centers):
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
        mask = labels == i
        ax.scatter(projected_data[mask, 0], projected_data[mask, 1], c=f'{i}', s=10)
        ax.scatter(projected_centers[i, 0], projected_centers[i, 1], c=f'{i}', s=100, marker='*', edgecolor='black')

        avg_distance = np.mean(np.linalg.norm(projected_data[mask] - projected_centers[i], axis=1))
        ax.text(
            projected_centers[i, 0],
            projected_centers[i, 1] + 2,
            f'Avg. distance: {avg_distance:.2f}',
            ha='center',
            fontsize=12,
        )

        min_distance = np.min(
            np.linalg.norm(projected_centers[np.arange(len(cluster_centers)) != i] - projected_centers[i], axis=1)
        )
        ax.text(
            projected_centers[i, 0],
            projected_centers[i, 1] - 8,
            f'Min. distance: {min_distance:.2f}',
            ha='center',
            fontsize=12,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

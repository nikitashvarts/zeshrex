import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def plot_loss_history(losses: List[float], output_file_path: os.PathLike) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.title('Training Loss')

    output_file_path = Path(output_file_path)
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file_path)

    plt.close()


def select_hard_negatives(embeddings, labels, device, margin=0.5, top_k=3):
    """
    Selects hard negatives within a batch based on cosine similarity.

    Args:
        embeddings (torch.Tensor): Tensor of shape [batch_size, embedding_dim].
        labels (torch.Tensor): Tensor of shape [batch_size], true labels for the batch.
        margin (float): Minimum margin for cosine similarity to qualify as hard negative.
        top_k (int): Number of hard negatives to select for each example.

    Returns:
        hard_negatives_indices (list): A list of lists, where each sublist contains
                                       indices of hard negatives for the corresponding
                                       batch element.
    """
    batch_size = embeddings.size(0)
    # Normalize embeddings for cosine similarity
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity between all pairs in the batch
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.T)

    hard_negatives_indices = []
    hard_negative_embeddings = []

    for i in range(batch_size):
        # Extract label and similarity scores for the current example
        current_label = labels[i]
        current_similarities = similarity_matrix[i]

        # Exclude self from similarity scores
        current_similarities[i] = -float('inf')  # Ensure self-similarity is not selected

        # Get indices of samples with different labels (negative examples)
        negative_mask = labels != current_label

        # Filter similarity scores for negative examples
        negative_similarities = current_similarities[negative_mask]
        negative_indices = torch.arange(batch_size).to(device)[negative_mask]

        # Select the top-k most similar negatives
        if len(negative_similarities) > 0:
            top_k_negatives = torch.topk(negative_similarities, min(top_k, len(negative_similarities))).indices
            selected_negatives = negative_indices[top_k_negatives].tolist()
        else:
            selected_negatives = []

        hard_negatives_indices.append(selected_negatives)
        hard_negative_embeddings.append(
            embeddings[selected_negatives] if selected_negatives else torch.empty(0, embeddings.size(1)).to(device)
        )

    hard_negatives_batch = torch.cat(hard_negative_embeddings, dim=0)

    return hard_negatives_batch

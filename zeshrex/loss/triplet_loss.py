from torch import nn


class TripletCosineMarginLoss:
    def __init__(self, margin: float = 7.5, alpha: float = 0.4):
        self._margin = margin
        self._alpha = alpha

    def __call__(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        cos = nn.CosineSimilarity()
        pos_similarity = cos(anchor_embeddings, positive_embeddings)
        neg_similarity = cos(anchor_embeddings, negative_embeddings)

        triplet_loss = nn.functional.relu(self._margin + neg_similarity - pos_similarity).mean()

        return triplet_loss


class TripletClassificationCosineMarginLoss:
    def __init__(self, margin: float = 7.5, alpha: float = 0.4):
        self._margin = margin
        self._alpha = alpha

        self._classification_criterion = nn.CrossEntropyLoss()

    def __call__(self, anchor_embeddings, positive_embeddings, negative_embeddings, logits, labels):
        cos = nn.CosineSimilarity()
        pos_similarity = cos(anchor_embeddings, positive_embeddings)
        neg_similarity = cos(anchor_embeddings, negative_embeddings)

        triplet_loss = nn.functional.relu(self._margin + neg_similarity - pos_similarity).mean()

        classification_loss = self._classification_criterion(logits, labels)

        loss = (1 - self._alpha) * (triplet_loss) + self._alpha * classification_loss

        return loss

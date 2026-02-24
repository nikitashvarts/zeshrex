from typing import Union

from .datasets import (
    RelationDataset,
    RelationTripletsDataset,
    RelationWithDescriptionDataset,
)

Dataset = Union[
    RelationDataset,
    RelationWithDescriptionDataset,
    RelationTripletsDataset,
]

__all__ = [
    Dataset,
    RelationDataset,
    RelationWithDescriptionDataset,
    RelationTripletsDataset,
]

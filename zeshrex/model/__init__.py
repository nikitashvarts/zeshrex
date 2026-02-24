from typing import Union

from .relation_classifier_model import RelationClassifierModel
from .relation_model import RelationModel
from .relation_triplets_model import RelationTripletsModel

Model = Union[RelationModel, RelationClassifierModel, RelationTripletsModel]


__all__ = [
    Model,
    RelationModel,
    RelationClassifierModel,
    RelationTripletsModel,
]

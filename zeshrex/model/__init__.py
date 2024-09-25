from typing import Union

from .relation_model import RelationClassifierModel, RelationModel

Model = Union[RelationModel, RelationClassifierModel]


__all__ = [RelationModel, RelationClassifierModel]

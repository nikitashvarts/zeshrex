import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from zeshrex.evaluation.common import measure_clusters_distances, visualize_clusters
from zeshrex.model import Model
from zeshrex.training.tools import select_hard_negatives


def eval_metric_model(
    model: Model,
    sentence_model: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    relations: Dict[str, int],
    criterion: nn.Module,
    output_dir: Optional[os.PathLike] = None,
    tag: Optional[str] = None,
):
    pass

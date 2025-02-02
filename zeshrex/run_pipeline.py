import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from zeshrex import CONFIG_FILE_PATH, PROJECT_PATH
from zeshrex.data.datasets import RelationDataset, RelationWithDescriptionDataset
from zeshrex.data.preprocessing import RelationTokenizationPreprocessor, SentenceTokenizationPreprocessor
from zeshrex.model import RelationClassifierModel, RelationTripletsModel
from zeshrex.training import (
    run_classification_training,
    run_triplet_classification_adaptive_training,
    run_triplet_training,
)
from zeshrex.utils import init_logger, load_yaml_config, print_configs


def run_pipeline(cfg: SimpleNamespace):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    relation_preprocessor = RelationTokenizationPreprocessor(
        tokenizer=tokenizer, max_len=cfg.dataset.max_len, relation_tokens=['<e1>', '</e1>', '<e2>', '</e2>']
    )

    dataset = RelationDataset.from_directory(
        dir_path=PROJECT_PATH / cfg.dataset.path, text_processor=relation_preprocessor
    )
    train_dataset, test_dataset, val_dataset = dataset.generate_train_test_split(
        use_predefined_split=cfg.dataset.use_predefined_split, use_zero_shot_split=cfg.dataset.use_zero_shot_split
    )

    # Model initialization
    # --------------------
    base_model = AutoModel.from_pretrained(cfg.model.name)

    # --------------------------------------------
    # Classification Model with Cross Entropy Loss
    # --------------------------------------------
    if cfg.train.criterion == 'CrossEntropyLoss':
        num_classes = len(train_dataset.labels)
        model = RelationClassifierModel(
            base_model=base_model,
            num_classes=num_classes,
            out_embedding_size=cfg.model.relation_embedding_size,
            dropout_rate=cfg.model.dropout_rate,
        )

        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        run_classification_training(cfg, model, train_dataset, test_dataset, device)

    # ------------------------------
    # Metric Model with Triplet Loss
    # ------------------------------
    elif cfg.train.criterion == 'TripletLoss':
        model = RelationTripletsModel(
            base_model=base_model,
            out_embedding_size=cfg.model.relation_embedding_size,
            dropout_rate=cfg.model.dropout_rate,
        )

        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        run_triplet_training(cfg, model, train_dataset, test_dataset, val_dataset, tokenizer, device)

    # ---------------------------------------------------------------
    # Metric + Classification Model with Triplet + Cross Entropy Loss
    # ---------------------------------------------------------------
    elif cfg.train.criterion == 'TripletClassificationLoss':
        sentence_preprocessor = SentenceTokenizationPreprocessor(tokenizer=tokenizer, max_len=cfg.dataset.max_len)

        train_rel_desc_dataset = RelationWithDescriptionDataset(
            train_dataset, desc_preprocessor=sentence_preprocessor
        )
        test_rel_desc_dataset = RelationWithDescriptionDataset(
            test_dataset, desc_preprocessor=sentence_preprocessor
        )

        sentence_model = AutoModel.from_pretrained(cfg.model.name)  # TODO: make a separate param
        for param in sentence_model.parameters():
            param.requires_grad = False  # NOTE: freeze all the parameters as the model is used for embeddings only

        num_classes = len(train_dataset.labels)
        model = RelationClassifierModel(
            base_model=base_model,
            num_classes=num_classes,
            out_embedding_size=cfg.model.relation_embedding_size,
            dropout_rate=cfg.model.dropout_rate,
        )

        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        sentence_model.to(device)

        run_triplet_classification_adaptive_training(
            cfg, model, sentence_model, train_rel_desc_dataset, test_rel_desc_dataset, device
        )

    else:
        raise Exception(f'Unknown criterion for training: {cfg.train.criterion}!')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    init_logger(file_name='run_pipeline.log', level=logging.INFO)
    # init_logger(file_name=None, level=logging.INFO)

    config: SimpleNamespace = load_yaml_config(CONFIG_FILE_PATH, convert_to_namespace=True)
    print_configs(config, print_function=logging.info)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.general.gpu)
    set_seed(config.general.seed)

    run_pipeline(config)

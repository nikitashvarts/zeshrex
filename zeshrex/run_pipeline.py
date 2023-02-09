import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig, get_linear_schedule_with_warmup, AdamW

from zeshrex import PROJECT_PATH, load_yaml_config, CONFIG_FILE_PATH, print_configs
from zeshrex.data.datasets import SemEval2010Task8Dataset
from zeshrex.data.preprocessing import RelationTokenizationPreprocessor
from zeshrex.model.relation_bert import RelationBert
from zeshrex.utils.logger import init_logger


def run_pipeline():
    # DATASETS
    # --------
    train_data_file_path = PROJECT_PATH / config.data.dataset_path / config.data.train_data_file_name

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model.name)
    text_preprocessor = RelationTokenizationPreprocessor(
        tokenizer=tokenizer, max_len=config.data.max_len, relation_tokens=["<e1>", "</e1>", "<e2>", "</e2>"]
    )

    train_dataset = SemEval2010Task8Dataset(data_path=train_data_file_path, text_processor=text_preprocessor)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
    )

    # PARAMETERS
    # ----------
    num_training_steps = \
        len(train_dataloader) // config.train.gradient_accumulation_steps * config.train.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    # -----
    bert_config = BertConfig.from_dict(config.model.__dict__)

    if config.model.use_pretrain:
        raise NotImplementedError('Model loading is not implemented! Please initialize a new one!')
    else:
        model = RelationBert(bert_config, output_size=config.model.output_size, dropout_rate=config.model.dropout_rate)

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # OPTIMIZER
    # ---------
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.train.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.train.learning_rate, eps=config.train.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.train.warmup_steps, num_training_steps=num_training_steps
    )

    # TRAINING
    # --------
    epoch_start = 1
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(epoch_start, int(config.train.num_epochs) + 1):
        train_loss = 0
        training_steps_count = 0
        with tqdm(total=len(train_dataloader) // config.train.gradient_accumulation_steps) as bar:
            for step, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_masks': batch[1],
                    'token_type_ids': batch[2],
                    'e1_masks': batch[3],
                    'e2_masks': batch[4],
                    'labels': batch[5]
                }

                loss = model(**inputs)

                loss = loss / config.train.gradient_accumulation_steps
                loss.backward()

                train_loss += loss.item()

    # test_data_file_path = PROJECT_PATH / config.data.dataset_path / config.data.test_data_file_name
    # test_dataset = SemEval2010Task8Dataset(test_data_file_path)
    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=...,
    #     shuffle=False,
    # )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    init_logger(file_name='run_pipeline.log', level=logging.DEBUG)  # TODO: remove debug

    config: SimpleNamespace = load_yaml_config(CONFIG_FILE_PATH, convert_to_namespace=True)
    print_configs(config, print_function=logging.info)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    set_seed(config.seed)

    run_pipeline()

import logging
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
)

from zeshrex import PROJECT_PATH, CONFIG_FILE_PATH
from zeshrex.data.datasets import RelationDataset
from zeshrex.data.preprocessing import RelationTokenizationPreprocessor
from zeshrex.evaluation import eval_model
from zeshrex.model.relation_bert import RelationBert
from zeshrex.utils import init_logger, print_configs, load_yaml_config


def run_pipeline():
    # DATASETS
    # --------
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model.name)
    text_preprocessor = RelationTokenizationPreprocessor(
        tokenizer=tokenizer, max_len=config.data.max_len, relation_tokens=["<e1>", "</e1>", "<e2>", "</e2>"]
    )

    dataset = RelationDataset.from_directory(
        dir_path=PROJECT_PATH / config.data.dataset_path, text_processor=text_preprocessor
    )

    train_dataset, test_dataset, val_dataset = dataset.generate_train_test_split()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_data,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_data,
    )

    # PARAMETERS
    # ----------
    num_training_steps = \
        len(train_dataloader) // config.train.gradient_accumulation_steps * config.train.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    # -----
    bert_config = BertConfig.from_dict(config.model.__dict__, **{'num_labels': len(train_dataset.labels)})

    if config.model.use_pretrain:
        raise NotImplementedError('Model loading is not implemented! Please initialize a new one!')
    else:
        model = RelationBert(bert_config, output_size=config.model.output_size, dropout_rate=config.model.dropout_rate)

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    state_save_path_template = str(PROJECT_PATH / config.models_output_dir / 'torch_model_{}_{}.bin')

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
    log_frequency = config.log_frequency // config.train.gradient_accumulation_steps
    eval_frequency = min(len(train_dataloader), config.train.eval_frequency)
    eval_frequency = eval_frequency // config.train.gradient_accumulation_steps

    epoch_start = 1
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(epoch_start, int(config.train.num_epochs) + 1):
        logging.info('========')
        logging.info(f'EPOCH {epoch}')
        logging.info('========')
        train_loss = 0
        training_steps_count = 0
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

            (loss, logits), embeddings = model(**inputs)

            loss = loss / config.train.gradient_accumulation_steps
            loss.backward()

            train_loss += loss.item()
            global_step += 1

            if (step + 1) % config.train.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                training_steps_count += 1

                if training_steps_count != 0 and training_steps_count % log_frequency == 0:
                    time.sleep(0.02)
                    logging.info(
                        'Epoch {:^3} Step {:^5} --- '
                        'Average loss (over {:^5} training steps): {:.5f} '
                        'LR: {}'.format(
                            epoch,
                            global_step,
                            training_steps_count,
                            train_loss / training_steps_count,
                            scheduler.get_last_lr(),
                        )
                    )

                if training_steps_count != 0 and training_steps_count % eval_frequency == 0:
                    logging.info('----------')
                    logging.info('Evaluation')
                    logging.info('----------')
                    val_result = eval_model(model, device, test_loader, relations=dataset.relations_encoding)
                    logging.info('Global step {:^5} VAL res: {}'.format(global_step, val_result))

                    if val_result['eval_loss'] < best_eval_loss:
                        best_eval_loss = val_result['eval_loss']
                        state_save_path = state_save_path_template.format(epoch, global_step)
                        logging.info(f"[Saving at] {state_save_path}")
                        torch.save(model.state_dict(), state_save_path)


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

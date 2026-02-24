import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch
from torch.utils.data import DataLoader

from zeshrex import PROJECT_PATH
from zeshrex.data import Dataset
from zeshrex.data.datasets import RelationTripletsDataset
from zeshrex.data.preprocessing import SentenceTokenizationPreprocessor
from zeshrex.evaluation.triplet import eval_metric_model
from zeshrex.loss.triplet_loss import TripletClassificationCosineMarginLoss
from zeshrex.model import Model
from zeshrex.training.tools import plot_loss_history


def run_triplet_training(
    cfg: SimpleNamespace,
    model: Model,
    train_dataset: RelationTripletsDataset,
    test_dataset: RelationTripletsDataset,
    tokenizer,  # TODO: define type
    device: torch.device,
):
    sentence_preprocessor = SentenceTokenizationPreprocessor(tokenizer=tokenizer, max_len=cfg.dataset.max_len)

    train_triplets_dataset = RelationTripletsDataset(
        train_dataset, triplets_per_sample=1, desc_preprocessor=sentence_preprocessor
    )
    test_triplets_dataset = RelationTripletsDataset(
        test_dataset, triplets_per_sample=1, desc_preprocessor=sentence_preprocessor
    )

    train_loader = DataLoader(
        dataset=train_triplets_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=RelationTripletsDataset.collate_data,
    )
    val_loader = DataLoader(  # TODO: replace with test_loader
        dataset=test_triplets_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        collate_fn=RelationTripletsDataset.collate_data,
    )

    # criterion = nn.TripletMarginLoss(margin=cfg.train.triplet_margin, p=2, eps=1e-7)
    # criterion = TripletCosineMarginLoss(margin=cfg.train.triplet_margin)
    criterion = TripletClassificationCosineMarginLoss(margin=cfg.train.triplet_margin)  # TODO: add alpha parameter
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    global_steps_count = 0
    steps_per_epoch = len(train_loader)

    losses: List[float] = []
    # Training loop
    # -------------
    for epoch in range(cfg.train.num_epochs):
        logging.info('========')
        logging.info(f'EPOCH {epoch + 1}')
        logging.info('========')

        running_loss: float = 0.0
        steps_count: int = 0
        for batch in train_loader:
            global_steps_count += 1
            steps_count += 1

            model.train()

            batch = tuple(t.to(device) for t in batch)

            inputs = {
                'anchor_input_ids': batch[0],
                'anchor_attention_mask': batch[1],
                'anchor_token_type_ids': batch[2],
                'anchor_e1_mask': batch[3],
                'anchor_e2_mask': batch[4],
                'pos_input_ids': batch[5],
                'pos_attention_mask': batch[6],
                # 'pos_token_type_ids': None,
                # 'pos_e1_mask': None,
                # 'pos_e2_mask': None,
                'pos_token_type_ids': batch[7],
                'pos_e1_mask': batch[8],
                'pos_e2_mask': batch[9],
                'neg_input_ids': batch[10],
                'neg_attention_mask': batch[11],
                'neg_token_type_ids': batch[12],
                'neg_e1_mask': batch[13],
                'neg_e2_mask': batch[14],
                # 'labels': batch[15],
                'desc_input_ids': batch[16],
                'desc_attention_mask': batch[17],
            }
            labels = batch[15]

            # anchor_embeddings, positive_embeddings, negative_embeddings = model(**inputs)
            # loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            anchor_embeddings, positive_embeddings, negative_embeddings, desc_embeddings, logits = model(**inputs)
            loss = criterion(anchor_embeddings, desc_embeddings, negative_embeddings, logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            avg_loss = running_loss / steps_count
            losses.append(avg_loss)

            if steps_count % cfg.general.log_frequency == 0:
                logging.info(
                    'Epoch {:^3} Step {:^5} --- '
                    'Average loss (over {:^5} training steps out of {}): {:.5f}'.format(
                        epoch + 1,
                        global_steps_count,
                        steps_count,
                        steps_per_epoch,
                        running_loss / steps_count,
                    )
                )

            losses_plot_file_name = 'loss_plot_{}.png'.format(Path(cfg.dataset.path).name.lower().replace(' ', '_'))
            losses_plot_file_path = PROJECT_PATH / cfg.general.output_dir / 'plots' / losses_plot_file_name
            plot_loss_history(losses=losses, output_file_path=losses_plot_file_path)

            # Calculate metrics on the validation set
            if steps_count % cfg.train.eval_frequency == 0 or steps_count % len(train_loader) == 0:
                val_result = eval_zero_shot_model(
                    model,
                    device,
                    val_loader,
                    relations=test_dataset.relations_encoding,
                    criterion=criterion,
                    output_dir=PROJECT_PATH / 'output' / 'viz',  # TODO: make a param
                    tag=f'{global_steps_count}steps',
                )
                logging.info(f"Epoch {epoch + 1}/{cfg.train.num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

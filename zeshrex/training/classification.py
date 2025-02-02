import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from zeshrex import PROJECT_PATH
from zeshrex.data import RelationDataset
from zeshrex.evaluation.classification import eval_classification_model
from zeshrex.model import Model
from zeshrex.training.tools import plot_loss_history


def run_classification_training(
    cfg: SimpleNamespace,
    model: Model,
    train_dataset: RelationDataset,
    test_dataset: RelationDataset,
    device: torch.device,
):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=RelationDataset.collate_data,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        collate_fn=RelationDataset.collate_data,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    global_steps_count = 0
    steps_per_epoch = len(train_loader)

    # Training loop
    # -------------
    for epoch in range(cfg.train.num_epochs):
        logging.info('========')
        logging.info(f'EPOCH {epoch + 1}')
        logging.info('========')

        losses: List[float] = []

        running_loss: float = 0.0
        steps_count: int = 0
        for batch in train_loader:
            global_steps_count += 1
            steps_count += 1

            model.train()

            batch = tuple(t.to(device) for t in batch)

            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'e1_mask': batch[3],
                'e2_mask': batch[4],
            }
            labels = batch[5]

            logits, _ = model(**inputs)
            loss = criterion(logits, labels)

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

            dataset_name = Path(cfg.dataset.path).name.lower().replace(' ', '_')
            losses_plot_file_name = 'loss_plot_{}_{}epoch.png'.format(dataset_name, epoch+1)
            losses_plot_file_path = PROJECT_PATH / cfg.general.output_dir / 'plots' / losses_plot_file_name
            plot_loss_history(losses=losses, output_file_path=losses_plot_file_path)

            # Calculate metrics on the validation set
            if steps_count % cfg.train.eval_frequency == 0 or steps_count % len(train_loader) == 0:
                metrics = eval_classification_model(
                    cfg=cfg,
                    model=model,
                    device=device,
                    dataloader=test_loader,
                    relations=test_dataset.relations_encoding,
                    criterion=criterion,
                    output_dir=PROJECT_PATH / 'output' / 'viz',  # TODO: make a param
                    tag=f'{dataset_name}_{global_steps_count}steps',
                )
                logging.info(f"Epoch {epoch + 1}/{cfg.train.num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
                logging.info(
                    'Validation Loss: {}, '
                    'Precision: {}, Recall: {}, F1-score: {}, '
                    'Avg Inner Dist: {}, Avg Outer Dict: {}'.format(
                        metrics['eval_loss'],
                        metrics['precision_macro'],
                        metrics['recall_macro'],
                        metrics['f1_score_macro'],
                        metrics['avg_inner_distance'],
                        metrics['avg_outer_distance'],
                    )
                )
                logging.info('==========')

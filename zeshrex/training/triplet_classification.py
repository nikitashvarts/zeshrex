import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from zeshrex import PROJECT_PATH
from zeshrex.data.datasets import RelationWithDescriptionDataset
from zeshrex.evaluation.triplet_classification import eval_metric_classification_model
from zeshrex.loss.triplet_loss import TripletClassificationCosineMarginLoss
from zeshrex.model import Model
from zeshrex.training.tools import plot_loss_history, select_hard_negatives


def run_triplet_classification_adaptive_training(
    cfg: SimpleNamespace,
    model: Model,
    sentence_model: nn.Module,
    train_dataset: RelationWithDescriptionDataset,
    test_dataset: RelationWithDescriptionDataset,
    device: torch.device,
):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=RelationWithDescriptionDataset.collate_data,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.train.eval_batch_size,
        shuffle=False,
        collate_fn=RelationWithDescriptionDataset.collate_data,
    )

    criterion = TripletClassificationCosineMarginLoss(margin=cfg.train.triplet_margin)  # TODO: add alpha parameter
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

            inputs_relation = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'e1_mask': batch[3],
                'e2_mask': batch[4],
            }
            labels = batch[5]
            inputs_description = {
                'input_ids': batch[6],
                'attention_mask': batch[7],
            }
            

            logits, anchor_embeddings = model(**inputs_relation)
            desc_embeddings = sentence_model(**inputs_description)[1]  # pooled output

            negative_embeddings = select_hard_negatives(anchor_embeddings, labels, device, margin=0.5, top_k=1)

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

            dataset_name = Path(cfg.dataset.path).name.lower().replace(' ', '_')
            losses_plot_file_name = 'loss_plot_{}_{}epoch.png'.format(dataset_name, epoch+1)
            losses_plot_file_path = PROJECT_PATH / cfg.general.output_dir / 'plots' / losses_plot_file_name
            plot_loss_history(losses=losses, output_file_path=losses_plot_file_path)

            # Calculate metrics on the validation set
            if steps_count % cfg.train.eval_frequency == 0 or steps_count % len(train_loader) == 0:
                metrics = eval_metric_classification_model(
                    cfg=cfg,
                    model=model,
                    sentence_model=sentence_model,
                    device=device,
                    dataloader=test_loader,
                    relation_labels=test_dataset.relations_encoding,  # TODO: make option for Generalized ZSL
                    relation_descriptions_tokens=test_dataset.relation_descriptions_tokens,
                    criterion=criterion,
                    use_zero_shot=cfg.dataset.use_zero_shot_split,
                    output_dir=PROJECT_PATH / 'output' / 'viz',  # TODO: make a param
                    tag=f'{dataset_name}_{global_steps_count}steps',
                )
                logging.info('-----------------------')
                logging.info(f"Epoch {epoch + 1}/{cfg.train.num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
                logging.info('-----------------------')
                logging.info('Metrics Report')
                logging.info('------------------------------------------------')
                logging.info('| {:^30} | {:^11} |'.format('Metric', 'Value'))
                logging.info('------------------------------------------------')
                for metric_name, metric_value in metrics.items():
                    logging.info('| {:^30} | {:^11.5f} |'.format(metric_name, metric_value))

                logging.info('------------------------------------------------')
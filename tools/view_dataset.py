import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

from matplotlib import pyplot as plt

from zeshrex import CONFIG_FILE_PATH, PROJECT_PATH
from zeshrex.data.datasets import RelationDataset
from zeshrex.utils import init_logger, load_yaml_config, print_configs


def load_args():
    parser = argparse.ArgumentParser('Dataset info viewer')
    parser.add_argument(
        '-o', '--output_dir', default='./output', help='Path to the directory to save output files like graphs, etc.'
    )
    return parser.parse_args()


def view_dataset(cfg: SimpleNamespace, output_dir: Optional[os.PathLike] = None) -> None:
    dataset = RelationDataset.from_directory(dir_path=PROJECT_PATH / cfg.dataset_path)

    relations_decoding_map: Dict[int, str] = {label: relation for relation, label in dataset.relations_encoding.items()}
    assert len(relations_decoding_map) == len(dataset.relations_encoding), 'Dataset contains duplicated labels!'

    relations_count = dict(Counter([relations_decoding_map[label] for _, label in dataset]))
    relations_count = {k: v for k, v in sorted(relations_count.items(), key=lambda item: item[1], reverse=False)}

    fig, ax = plt.subplots(figsize=(15, 7))
    # fig.subplots_adjust(left=0.15)
    ax.barh(range(len(relations_count)), list(relations_count.values()), align='center')
    ax.set_yticks(range(len(relations_count)), list(relations_count.keys()))
    ax.set_xlabel('Number of samples')
    ax.set_title(f'{Path(cfg.dataset_path).stem}')
    # ax.bar_label(bars)
    # ax.set_xticks(range(len(relations_count)), list(relations_count.keys()), rotation='vertical')

    if output_dir:
        output_dir = PROJECT_PATH / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'relations_count.png')
    else:
        logging.warning('Output dir is not specified! Results will not be saved!')

    logging.info('Done!')


if __name__ == '__main__':
    init_logger()
    args = load_args()

    config: SimpleNamespace = load_yaml_config(CONFIG_FILE_PATH, convert_to_namespace=True)
    dataset_config = config.data
    print_configs(dataset_config, print_function=logging.info)

    view_dataset(dataset_config, args.output_dir)

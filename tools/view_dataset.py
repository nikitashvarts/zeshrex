import argparse
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

from matplotlib import pyplot as plt

from zeshrex import PROJECT_PATH
from zeshrex.data.datasets import RelationDataset
from zeshrex.utils import init_logger


def load_args():
    parser = argparse.ArgumentParser('Dataset info viewer')
    parser.add_argument(
        '-d',
        '--dataset_path',
        type=str,
        default='./datasets/prepared/WebNLG/',
        help='Path to the directory containing the dataset files.',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        default='./output/datasets_info/',
        help='Path to the directory to save output files like graphs, etc.',
    )
    return parser.parse_args()


def view_dataset(dataset_path: os.PathLike, output_dir: Optional[os.PathLike] = None) -> None:
    dataset = RelationDataset.from_directory(dir_path=PROJECT_PATH / dataset_path)
    dataset_name = Path(dataset_path).stem

    relations_decoding_map: Dict[int, str] = {label: relation for relation, label in dataset.relations_encoding.items()}
    assert len(relations_decoding_map) == len(dataset.relations_encoding), 'Dataset contains duplicated labels!'

    relations_count = dict(Counter([relations_decoding_map[label] for _, label in dataset]))
    relations_count = {k: v for k, v in sorted(relations_count.items(), key=lambda item: item[1], reverse=False)}

    fig, ax = plt.subplots(figsize=(15, 7))
    # fig.subplots_adjust(left=0.15)
    ax.barh(range(len(relations_count)), list(relations_count.values()), align='center')
    ax.set_yticks(range(len(relations_count)), list(relations_count.keys()))
    ax.set_xlabel('Number of samples')
    ax.set_title(f'{dataset_name}')
    # ax.bar_label(bars)
    # ax.set_xticks(range(len(relations_count)), list(relations_count.keys()), rotation='vertical')

    if output_dir:
        output_dir = PROJECT_PATH / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'relations_count_{dataset_name}.png')
    else:
        logging.warning('Output dir is not specified! Results will not be saved!')

    logging.info('Done!')


if __name__ == '__main__':
    init_logger()
    args = load_args()

    view_dataset(args.dataset_path, args.output_dir)

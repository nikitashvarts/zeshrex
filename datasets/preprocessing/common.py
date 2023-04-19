import csv
import logging
import os
from pathlib import Path
from typing import List, Dict, Any


def load_relation_names(file_path: os.PathLike) -> List[str]:
    logging.info(f'Taking relation names from {file_path}')
    with open(file_path, 'r') as f:
        relation_names = [line.strip() for line in f.readlines()]
    return relation_names


def save_data(data: List[Dict[str, Any]], dir_path: os.PathLike) -> None:
    dir_path = Path(dir_path)
    logging.info(f'Saving data to {dir_path}')

    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / 'data.tsv'

    fieldnames = list(data[0].keys())
    with open(file_path, 'w') as tsv_file:
        writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for sample in data:
            writer.writerow(sample)


def save_index(index: List[int], dir_path: os.PathLike, name: str) -> None:
    dir_path = Path(dir_path)
    logging.info(f'Saving {name} index to {dir_path}')

    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f'{name}_index.txt'

    with open(file_path, 'w') as txt_file:
        txt_file.writelines([f'{i}\n' for i in index])

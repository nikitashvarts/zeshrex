import csv
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Iterable, Optional, Set

import torch
from torch.utils.data import Dataset

from zeshrex.data.preprocessing import BasePreprocessor

Data = List[Dict[str, Any]]
DataIndex = Optional[List[int]]


class RelationDataset(Dataset):
    def __init__(
            self,
            data: Data,
            relations: List[str],
            indexes: Optional[Tuple[DataIndex, DataIndex, DataIndex]] = None,
            text_processor: Optional[BasePreprocessor] = None,
    ) -> None:
        self._dataset = data

        self._text_processor = text_processor
        if self._text_processor is None:
            logging.warning('Text processor is not specified! Dataset will output raw text data!')

        self._train_index = indexes[0] if indexes else None
        self._test_index = indexes[1] if indexes else None
        self._val_index = indexes[2] if indexes else None

        assert len(relations) == len(set(relations)), 'Provided relation names must be unique!'
        self._relation_to_label: Dict[str, int] = self._encode_relations(relations)

        logging.info(f'Dataset was initialized with the following relations: {relations}')

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[Tuple[Iterable, ...], int]:
        item = self._dataset[index]
        if self._text_processor:
            preprocessed_data: Tuple[Iterable, ...] = self._text_processor(item['text'])
        else:
            preprocessed_data = item['text']
        return preprocessed_data, self._relation_to_label[item['relation']]

    @classmethod
    def from_directory(
            cls,
            dir_path: os.PathLike,
            text_processor: Optional[BasePreprocessor] = None
    ) -> 'RelationDataset':
        _DATA_FILE_NAME = 'data.tsv'
        _INDEX_FILE_NAME_TEMPLATE = '{}_index.txt'

        dir_path = Path(dir_path)

        data, relations = cls._load_data(dir_path / _DATA_FILE_NAME)
        train_index = cls._load_index(dir_path / _INDEX_FILE_NAME_TEMPLATE.format('train'))
        test_index = cls._load_index(dir_path / _INDEX_FILE_NAME_TEMPLATE.format('test'))
        val_index = cls._load_index(dir_path / _INDEX_FILE_NAME_TEMPLATE.format('val'))

        assert len(data) == sum([len(idx) if idx is not None else 0 for idx in (train_index, test_index, val_index)]), \
            "Lengths of data and all indexes are not equal! Check consistency of your data!"

        return cls(
            data=data, indexes=(train_index, test_index, val_index), relations=relations, text_processor=text_processor
        )

    @property
    def labels(self) -> Optional[List[int]]:
        if self._relation_to_label is None:
            logging.warning('Relation have not been defined yet!')
            return None
        return list(self._relation_to_label.values())

    @property
    def relations_encoding(self) -> Optional[Dict[str, int]]:
        if self._relation_to_label is None:
            logging.warning('Relation have not been defined yet!')
            return None
        return self._relation_to_label.copy()

    def generate_train_test_split(
            self, use_predefined_split: bool = True
    ) -> Tuple['RelationDataset', 'RelationDataset', 'RelationDataset']:
        relations = list(self._relation_to_label.keys())
        if use_predefined_split:
            split_data: Dict[str, Data] = {}
            for sample in self._dataset:
                if self._train_index and sample['index'] in self._train_index:
                    split_data['train'] = split_data.get('train', []) + [sample]
                    continue

                if self._test_index and sample['index'] in self._test_index:
                    split_data['test'] = split_data.get('test', []) + [sample]
                    continue

                if self._val_index and sample['index'] in self._val_index:
                    split_data['val'] = split_data.get('val', []) + [sample]
                    continue

            split = (
                RelationDataset(
                    split_data.get('train', {}), relations, (self._train_index, None, None), self._text_processor,
                ),
                RelationDataset(
                    split_data.get('test', {}), relations, (None, self._test_index, None), self._text_processor,
                ),
                RelationDataset(
                    split_data.get('val', {}), relations, (None, None, self._val_index), self._text_processor,
                ),
            )

            return split
        else:
            raise NotImplementedError('Random or zero-shot split is not defined yet!')

    @staticmethod
    def collate_data(batch: List[Tuple[Tuple[Iterable, ...], int]]) -> List[torch.Tensor]:
        collated_data: Dict[int, List[List[Any]]] = {}
        collated_labels: List[int] = []
        for data, label in batch:
            for index, item in enumerate(data):
                collated_data[index] = collated_data.get(index, []) + [list(item)]
            collated_labels.append(label)

        collated_tensors: List[torch.Tensor, ...] = []
        for collated_items in collated_data.values():
            collated_tensors.append(torch.tensor(collated_items, dtype=torch.long))

        collated_tensors.append(torch.tensor(collated_labels, dtype=torch.long))

        return collated_tensors

    @staticmethod
    def _load_data(file_path: os.PathLike) -> Tuple[Data, List[str]]:
        file_path = Path(file_path)
        logging.info(f'Loading data from {file_path}')
        assert file_path.exists(), 'Specified path does not exist!'

        data: Data = []
        relations: Set[str] = set()
        with open(file_path) as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')
            for row in reader:
                sample = {'index': int(row['index']), 'relation': row['relation'], 'text': row['text']}
                data.append(sample)
                relations.add(row['relation'])

        return data, sorted(relations)

    @staticmethod
    def _load_index(file_path: os.PathLike) -> Optional[List[int]]:
        file_path = Path(file_path)
        logging.info(f'Loading index from {file_path}')
        if not file_path.exists():
            logging.info(f'{file_path.name} not found! Skipping...')
            return None

        with open(file_path) as f:
            index: List[int] = [int(i) for i in f.readlines()]

        return index

    @staticmethod
    def _encode_relations(relations: List[str]) -> Dict[str, int]:
        assert len(relations) == len(set(relations)), 'Provided relation names must be unique!'
        relation_to_label: Dict[str, int] = {rel: index for index, rel in enumerate(relations)}
        return relation_to_label

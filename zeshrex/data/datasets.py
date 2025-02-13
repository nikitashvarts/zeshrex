import csv
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.preprocessing.common import load_relation_names
from zeshrex import PROJECT_PATH
from zeshrex.data.preprocessing import BasePreprocessor

Sample = Dict[str, Any]
Data = List[Sample]
DataIndex = Optional[List[int]]


class RelationDataset(Dataset):
    def __init__(
        self,
        data: Data,
        dataset_name: str,
        relations: List[str],
        indexes: Optional[Tuple[DataIndex, DataIndex, DataIndex]] = None,
        text_processor: Optional[BasePreprocessor] = None,
        limit: Optional[int] = None,  # TODO: remove debug
    ) -> None:
        self._dataset = data
        self._dataset_name = dataset_name

        if limit is not None:  # TODO: remove debug
            self._dataset = data[:limit]

        self._text_processor = text_processor
        if self._text_processor is None:
            logging.warning('Text processor is not specified! Dataset will output raw text data!')

        self._train_index = indexes[0] if indexes else None
        self._test_index = indexes[1] if indexes else None
        self._val_index = indexes[2] if indexes else None

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
        cls, dir_path: os.PathLike, text_processor: Optional[BasePreprocessor] = None
    ) -> 'RelationDataset':
        _DATA_FILE_NAME = 'data.tsv'
        _INDEX_FILE_NAME_TEMPLATE = '{}_index.txt'

        dir_path = Path(dir_path)

        data, relations = cls._load_data(dir_path / _DATA_FILE_NAME)
        train_index = cls._load_index(dir_path / _INDEX_FILE_NAME_TEMPLATE.format('train'))
        test_index = cls._load_index(dir_path / _INDEX_FILE_NAME_TEMPLATE.format('test'))
        val_index = cls._load_index(dir_path / _INDEX_FILE_NAME_TEMPLATE.format('val'))

        assert len(data) == sum(
            [len(idx) if idx is not None else 0 for idx in (train_index, test_index, val_index)]
        ), 'Lengths of data and all indexes are not equal! Check consistency of your data!'

        dataset_name = Path(dir_path).name

        return cls(
            data=data,
            dataset_name=dataset_name,
            indexes=(train_index, test_index, val_index),
            relations=relations,
            text_processor=text_processor,
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
        self,
        use_predefined_split: bool = False,
        use_zero_shot_split: bool = False,
    ) -> Tuple['RelationDataset', 'RelationDataset', 'RelationDataset']:
        if use_predefined_split:
            logging.info('Generating split according to provided train / test / val indexes')
            return self._generate_predefined_split()
        elif use_zero_shot_split:
            logging.info('Generating random Zero-Shot split')
            return self._generate_zero_shot_split()  # TODO: support unseen_classes_ratio arg
        else:
            logging.info('Generating simple random train / test / val split')
            raise NotImplementedError('Simple random split is not defined yet!')  # TODO: implement random split

    @staticmethod
    def collate_data(batch: List[Tuple[Tuple[Iterable, ...], int]]) -> List[torch.Tensor]:
        collated_data: Dict[int, List[List[Any]]] = {}
        collated_labels: List[int] = []
        for data, label in batch:
            for index, item in enumerate(data):
                collated_data[index] = collated_data.get(index, []) + [list(item)]
            collated_labels.append(label)

        collated_tensors: List[torch.Tensor] = []
        for collated_items in collated_data.values():
            collated_tensors.append(torch.tensor(collated_items, dtype=torch.long))

        collated_tensors.append(torch.tensor(collated_labels, dtype=torch.long))

        return collated_tensors

    def _generate_predefined_split(self):
        relations = list(self._relation_to_label.keys())
        split_data: Dict[str, Data] = {}
        for sample in self._dataset:
            if self._train_index is not None:
                if sample['index'] in self._train_index:
                    split_data['train'] = split_data.get('train', []) + [sample]
                    continue

            if self._test_index is not None:
                if sample['index'] in self._test_index:
                    split_data['test'] = split_data.get('test', []) + [sample]
                    continue

            if self._val_index is not None:
                if sample['index'] in self._val_index:
                    split_data['val'] = split_data.get('val', []) + [sample]
                    continue
            logging.warning('Skipping element as its index not presented in any predefined index set!')

        split = (
            RelationDataset(
                data=split_data.get('train', []),
                dataset_name=self._dataset_name,
                relations=relations,
                indexes=(self._train_index, None, None),
                text_processor=self._text_processor,
                # limit=1000,  # TODO: remove debug
            ),
            RelationDataset(
                data=split_data.get('test', []),
                dataset_name=self._dataset_name,
                relations=relations,
                indexes=(None, self._test_index, None),
                text_processor=self._text_processor,
                # limit=300,  # TODO: remove debug
            ),
            RelationDataset(
                data=split_data.get('val', []),
                dataset_name=self._dataset_name,
                relations=relations,
                indexes=(None, None, self._val_index),
                text_processor=self._text_processor,
                # limit=300,  # TODO: remove debug
            ),
        )

        return split

    def _generate_zero_shot_split(self, unseen_classes_ratio: float = 0.3, seed: Optional[int] = None):
        relations = list(self._relation_to_label.keys())
        
        rng = np.random.RandomState(seed)  # create a separate state not to break the main one for reproducibility
        unseen_relations = set(rng.choice(relations, size=int(len(relations) * unseen_classes_ratio), replace=False))
        seen_relations = set(rel for rel in relations if rel not in unseen_relations)

        # WebNLG Predefined
        unseen_relations = set(['ethnicGroup', 'creator', 'language', 'number'])
        
        # NYT Predefined
        # unseen_relations = set(['country', 'company', 'place_lived'])

        # NEREL Predefined
        # unseen_relations = set(['PART_OF', 'AGE_IS', 'LOCATED_IN', 'POINT_IN_TIME'])

        seen_relations = set(rel for rel in relations if rel not in unseen_relations)

        split_data: Dict[str, Data] = {}
        split_indexes: Dict[str, List[int]] = {}

        for sample in self._dataset:
            if sample['relation'] in seen_relations:
                split_data['train'] = split_data.get('train', []) + [sample]
                split_indexes['train'] = split_indexes.get('train', []) + [sample['index']]
                continue

            if sample['relation'] in unseen_relations:
                split_data['test'] = split_data.get('test', []) + [sample]
                split_indexes['test'] = split_indexes.get('test', []) + [sample['index']]
                continue

            logging.warning('Skipping element as its index not presented in any relations set!')

        split = (
            RelationDataset(
                data=split_data.get('train', []),
                dataset_name=self._dataset_name,
                relations=list(seen_relations),
                indexes=(split_indexes.get('train', []), None, None),
                text_processor=self._text_processor,
                # limit=1000,  # TODO: remove debug
            ),
            RelationDataset(
                data=split_data.get('test', []),
                dataset_name=self._dataset_name,
                relations=list(unseen_relations),
                indexes=(None, split_indexes.get('test', []), None),
                text_processor=self._text_processor,
                # limit=300,  # TODO: remove debug
            ),
            RelationDataset(
                data=split_data.get('val', []),
                dataset_name=self._dataset_name,
                relations=[],  # TODO: support relations for validation
                indexes=(None, None, []),
                text_processor=self._text_processor,
                # limit=300,  # TODO: remove debug
            ),
        )

        return split

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
                row: Dict
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


class RelationWithDescriptionDataset(Dataset):
    def __init__(
        self,
        data: RelationDataset,
        desc_preprocessor: Optional[BasePreprocessor] = None,
    ):
        self._data = data
        self._dataset_name = data._dataset_name
        self._desc_preprocessor = desc_preprocessor

        self._label_to_relation = {v: k for k, v in self._data._relation_to_label.items()}

        self._relation_to_desc = load_relation_names(
            PROJECT_PATH / 'datasets' / 'prepared' / self._dataset_name / 'relation_names.tsv'
        )
        self._relation_to_desc = {
            rel: desc for rel, desc in self._relation_to_desc.items() if rel in list(self._label_to_relation.values())
        }
        self._relation_to_desc_tokens = self._preprocess_descriptions(self._relation_to_desc)

        self._dataset = self._connect_samples_with_description(data)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Any, Any], int, Any]:
        return self._dataset[index]

    @property
    def relations_encoding(self) -> Optional[Dict[str, int]]:
        if self._data._relation_to_label is None:
            logging.warning('Relation have not been defined yet!')
            return None
        return self._data._relation_to_label.copy()

    @property
    def relation_descriptions_tokens(self) -> Optional[Dict[str, Tuple[List[int], List[int]]]]:
        if self._relation_to_desc_tokens is None:
            logging.warning('Relation have not been defined yet!')
            return None
        return self._relation_to_desc_tokens.copy()

    @staticmethod
    def collate_data(batch) -> List[torch.Tensor]:
        collated_data = {}
        collated_labels = []
        collated_desc = {}
        for data, label, desc in batch:
            for index, item in enumerate(data):
                collated_data[index] = collated_data.get(index, []) + [list(item)]
            collated_labels.append(label)
            for index, item in enumerate(desc):
                collated_desc[index] = collated_desc.get(index, []) + [list(item)]

        collated_tensors: List[torch.Tensor] = []
        for collated_items in collated_data.values():
            collated_tensors.append(torch.tensor(collated_items, dtype=torch.long))

        collated_tensors.append(torch.tensor(collated_labels, dtype=torch.long))

        for collated_items in collated_desc.values():
            collated_tensors.append(torch.tensor(collated_items, dtype=torch.long))

        return collated_tensors

    def _preprocess_descriptions(self, relation_to_desc: Dict[str, str]) -> Dict[str, Tuple[List[int], List[int]]]:
        return {
            relation: self._desc_preprocessor(desc_text)
            for relation, desc_text in relation_to_desc.items()
        }
        
    def _connect_samples_with_description(
        self,
        data: RelationDataset,
    ):
        logging.info('Connecting samples with descriptions')
        samples_with_desc: List[Tuple[Any, Any, Any]] = []

        for sample, relation in tqdm(data):
            desc_tokens = self._relation_to_desc_tokens[self._label_to_relation[relation]]
            samples_with_desc.append((sample, relation, desc_tokens))

        return samples_with_desc


class RelationTripletsDataset(Dataset):
    def __init__(
        self,
        data: RelationDataset,
        triplets_per_sample: int = 5,
        desc_preprocessor: Optional[BasePreprocessor] = None,
    ) -> None:
        self._data = data
        self._triplets_per_sample = triplets_per_sample

        self.relation_to_desc = load_relation_names(
            PROJECT_PATH / 'datasets' / 'prepared' / self._dataset_name / 'relation_names.tsv'
        )

        self.label_to_relation = {v: k for k, v in self._data._relation_to_label.items()}

        self._desc_preprocessor = desc_preprocessor
        self._dataset = self._make_positive_negative_triplets(data)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Any, Any], int, Any]:
        return self._dataset[index]

    @staticmethod
    def collate_data(batch: List[Tuple[Tuple[Iterable, ...], int]]) -> List[torch.Tensor]:
        collated_data: Dict[int, List[List[Any]]] = {}
        collated_labels: List[int] = []
        collated_desc: Dict[int, Any] = {}
        for data, label, desc in batch:
            for index, item in enumerate(data):
                collated_data[index] = collated_data.get(index, []) + [list(item)]
            collated_labels.append(label)
            for index, item in enumerate(desc):
                collated_desc[index] = collated_desc.get(index, []) + [list(item)]

        collated_tensors: List[torch.Tensor] = []
        for collated_items in collated_data.values():
            collated_tensors.append(torch.tensor(collated_items, dtype=torch.long))

        collated_tensors.append(torch.tensor(collated_labels, dtype=torch.long))

        for collated_items in collated_desc.values():
            collated_tensors.append(torch.tensor(collated_items, dtype=torch.long))

        return collated_tensors

    def _make_positive_negative_triplets(
        self,
        data: RelationDataset,
    ) -> List[Tuple[Tuple[Any, Any, Any], int, Any]]:
        logging.info('Making positive and negative triplets from data')
        triplets: List[Tuple[Tuple[Any, Any, Any], int, Any]] = []
        pbar = tqdm(total=len(data), miniters=1)
        for anchor_sample, anchor_relation in data:

            # -------------------------------------------------------
            desc_sample_text = self.relation_to_desc[self.label_to_relation[anchor_relation]]
            desc_sample = self._desc_preprocessor(desc_sample_text)
            # -------------------------------------------------------

            for i in range(self._triplets_per_sample):
                while True:
                    positive_sample, positive_relation = random.choice(data)
                    if positive_sample == anchor_sample:
                        continue
                    if positive_relation == anchor_relation:
                        break
                while True:
                    negative_sample, negative_relation = random.choice(data)
                    if negative_relation != anchor_relation:
                        break
                #: TODO: refactor (desc_sample instead of positive)
                triplets.append(((*anchor_sample, *positive_sample, *negative_sample), anchor_relation, desc_sample))
            pbar.update(1)
        pbar.close()

        assert len(triplets) == self._triplets_per_sample * len(data), 'Wrong amount of generated triplets!'
        return triplets

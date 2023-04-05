import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any, Iterable, Optional

import torch
from torch.utils.data import Dataset

from zeshrex.data.preprocessing import BasePreprocessor


class BaseRelationDataset(Dataset):
    def __init__(
            self, data_path: Union[str, Path],
            text_processor: BasePreprocessor,
            relation_names_file_path: Optional[os.PathLike] = None,
            *args,
            **kwargs,
    ):
        self._data_file_path = Path(data_path)
        assert self._data_file_path.exists(), f'Data file {self._data_file_path} was not found!'

        self._relation_names: Optional[List[str]] = self._load_relation_names(relation_names_file_path)
        self._dataset: List[Dict[str, Any]] = self._load_data(self._data_file_path)
        self._text_processor = text_processor

        self._relation_to_label: Optional[Dict[str, int]] = self._encode_relations()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[Tuple[Iterable, ...], int]:
        item = self._dataset[index]
        preprocessed_data: Tuple[Iterable, ...] = self._text_processor(item['text'])
        return preprocessed_data, self._relation_to_label[item['relation']]

    @property
    @abstractmethod
    def labels(self) -> List[Any]:
        pass

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

    @abstractmethod
    def _load_data(self, data_path: os.PathLike) -> List[Dict[str, Any]]:
        pass

    def _encode_relations(self) -> Dict[str, int]:
        assert self._relation_names is not None, 'Relation names are not defined!'
        relation_to_label: Dict[str, int] = {rel: index for index, rel in enumerate(self._relation_names)}
        return relation_to_label

    @staticmethod
    def _load_relation_names(file_path: Optional[os.PathLike]) -> Optional[List[str]]:
        if file_path is not None:
            try:
                logging.info(f'Taking relation names from {file_path}')
                with open(file_path, 'r') as f:
                    relation_names = [line.strip() for line in f.readlines()]
                return relation_names
            except Exception as e:
                logging.error('Cannot read file with relation names! Relations will be derived from data.')
                logging.exception(e)
                return None
        else:
            logging.info('Relation names are not provided and will be derived from data')
            return None


class SemEval2010Task8Dataset(BaseRelationDataset):
    @property
    def labels(self) -> Optional[List[int]]:
        if self._relation_to_label is None:
            logging.warning('Relation names have not been defined yet!')
        return list(self._relation_to_label.values())

    def _load_data(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        raw_data_list = []
        with open(data_path, 'r') as tf:
            counter = 0
            current_text_chunk = []
            for line in tf:
                if counter == 3:
                    raw_data_list.append(current_text_chunk)
                    counter = 0
                    current_text_chunk = []
                else:
                    current_text_chunk.append(line)
                    counter += 1

        data: List[Dict[str, Any]] = []
        relations: List[str] = []
        for chunk in raw_data_list:
            raw_id_and_text, raw_relation_type, raw_comment = chunk
            text_id = int(raw_id_and_text.split('\t')[0])
            text = raw_id_and_text.split('\t')[-1].strip('\n').strip('\"')
            relation_name = raw_relation_type.strip('\n')
            comment = raw_comment.strip('\n').split(':', 1)[-1].strip()
            if self._relation_names is not None:
                assert relation_name in self._relation_names, f'Unknown relation {relation_name}!'
            else:
                relations.append(relation_name)
            data.append({'id': text_id, 'text': text, 'relation': relation_name, 'comment': comment})

        if self._relation_names is None:
            self._relation_names = list(set(relations))

        return data

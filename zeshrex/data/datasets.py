from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any

from torch.utils.data import Dataset

from zeshrex.data.preprocessing import BasePreprocessor


class BaseDataset(Dataset):
    def __init__(self, data_path: Union[str, Path], text_processor: BasePreprocessor):
        self._data_file_path = Path(data_path)
        assert self._data_file_path.exists(), f'Data file {self._data_file_path} was not found!'
        self._dataset: List[Dict[str, Any]] = self._load_data(self._data_file_path)
        self._text_processor = text_processor

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        item = self._dataset[index]
        preprocessed_text = self._text_processor(item['text'])
        return preprocessed_text, item['label']

    @abstractmethod
    def _load_data(self, data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        pass


class SemEval2010Task8Dataset(BaseDataset):
    _RELATION_TYPES = (
        "Cause-Effect(e1,e2)",
        "Cause-Effect(e2,e1)",
        "Component-Whole(e1,e2)",
        "Component-Whole(e2,e1)",
        "Content-Container(e1,e2)",
        "Content-Container(e2,e1)",
        "Entity-Destination(e1,e2)",
        "Entity-Destination(e2,e1)",
        "Entity-Origin(e1,e2)",
        "Entity-Origin(e2,e1)",
        "Instrument-Agency(e1,e2)",
        "Instrument-Agency(e2,e1)",
        "Member-Collection(e1,e2)",
        "Member-Collection(e2,e1)",
        "Message-Topic(e1,e2)",
        "Message-Topic(e2,e1)",
        "Product-Producer(e1,e2)",
        "Product-Producer(e2,e1)",
        "Other",
    )

    _TYPE_TO_LABEL = {rel_type: index for index, rel_type in enumerate(_RELATION_TYPES)}

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
        for chunk in raw_data_list:
            raw_id_and_text, raw_relation_type, raw_comment = chunk
            text_id = int(raw_id_and_text.split('\t')[0])
            text = raw_id_and_text.split('\t')[-1].strip('\n').strip('\"')
            relation_type = raw_relation_type.strip('\n')
            comment = raw_comment.strip('\n').split(':', 1)[-1].strip()
            assert relation_type in self._RELATION_TYPES, f'Unknown type {relation_type}!'
            data.append(
                {'id': text_id, 'text': text, 'label': self._TYPE_TO_LABEL[relation_type], 'comment': comment}
            )
        return data

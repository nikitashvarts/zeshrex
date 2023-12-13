import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from datasets.preprocessing.common import save_data, save_index, strip_accents, load_relation_names
from zeshrex import PROJECT_PATH
from zeshrex.utils import init_logger, print_configs


def load_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser('Script for NYT dataset preparation')

    parser.add_argument('--data_dir', type=str, default='./datasets/raw/NYT/')
    parser.add_argument('--train_file', type=str, default='./train.json')
    parser.add_argument('--test_file', type=str, default='./test.json')
    parser.add_argument('--val_file', type=str, default='./valid.json')
    parser.add_argument('--relation_names_file', type=str, default='./relation_names_top.txt')
    parser.add_argument('--output_dir', type=str, default='./datasets/prepared/NYT/')

    return parser.parse_args().__dict__


def load_data(
    file_path: os.PathLike,
    relation_names: Optional[List[str]] = None,
    initial_index: int = 1,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    file_path = Path(file_path)
    assert file_path.suffix == '.json', 'Data file must be a JSON file for NYT dataset'
    raw_data: List[Dict[str, Any]] = [json.loads(line) for line in open(file_path, 'r')]

    if relation_names is not None:
        logging.info(f'Loading data with the following relations: {relation_names}')
        logging.info('NOTE that relations not included in this list will be skipped!')
        relation_names_set = set(relation_names)
        assert len(relation_names_set) == len(relation_names), 'Duplicated relations found!'
    else:
        relation_names_set = set()

    data: List[Dict[str, Any]] = []
    indexes: List[int] = []
    derived_relations: List[str] = []
    current_index = initial_index
    for sample_data in raw_data:
        raw_text: str = sample_data['sentText']

        for single_relation_data in sample_data['relationMentions']:
            relation = single_relation_data['label'].rsplit('/', 1)[-1]
            if relation_names is not None and relation not in relation_names_set:
                logging.debug(f'Skipping relation {relation} as it is not stated in relation names!')
                continue
            derived_relations.append(relation)

            sub_text = strip_accents(single_relation_data['em1Text'])
            obj_text = strip_accents(single_relation_data['em2Text'])
            try:
                sub_begin_idx = raw_text.index(sub_text)
                obj_begin_idx = raw_text.index(obj_text)
            except ValueError:
                continue

            if sub_begin_idx <= obj_begin_idx:
                first_text, second_text = sub_text, obj_text
                first_beg_idx, second_beg_idx = sub_begin_idx, obj_begin_idx
                first_beg_token, first_end_token = '<e1>', '</e1>'
                second_beg_token, second_end_token = '<e2>', '</e2>'
            else:
                first_text, second_text = obj_text, sub_text
                first_beg_idx, second_beg_idx = obj_begin_idx, sub_begin_idx
                first_beg_token, first_end_token = '<e2>', '</e2>'
                second_beg_token, second_end_token = '<e1>', '</e1>'

            text = raw_text[:first_beg_idx]
            text += first_beg_token + raw_text[first_beg_idx: (first_beg_idx + len(first_text))] + first_end_token
            text += raw_text[(first_beg_idx + len(first_text)): second_beg_idx]
            text += second_beg_token + raw_text[second_beg_idx: (second_beg_idx + len(second_text))] + second_end_token
            text += raw_text[(second_beg_idx + len(second_text)):]

            sample = {
                'index': current_index,
                'relation': relation,
                'text': text,
            }
            data.append(sample)
            indexes.append(current_index)
            current_index += 1

    if relation_names is None:
        relation_names = list(set(derived_relations))

    logging.info(f'Data were collected including the following relations: {relation_names}')

    return data, indexes


def main(args: Dict[str, Any]) -> None:
    dataset_path = PROJECT_PATH / args['data_dir']
    output_path = PROJECT_PATH / args['output_dir']

    relation_names_file_path = (dataset_path / args['relation_names_file']) if args['relation_names_file'] else None
    relation_names = load_relation_names(relation_names_file_path) if relation_names_file_path else None

    train_data_path = dataset_path / args['train_file']
    test_data_path = dataset_path / args['test_file']
    val_data_path = dataset_path / args['val_file']

    train_data, train_index = load_data(train_data_path, relation_names)
    test_data, test_index = load_data(test_data_path, relation_names, initial_index=max(train_index) + 1)
    val_data, val_index = load_data(val_data_path, relation_names, initial_index=max(test_index) + 1)

    joined_data = [*train_data, *test_data, *val_data]

    save_data(joined_data, output_path)
    save_index(train_index, output_path, 'train')
    save_index(test_index, output_path, 'test')
    save_index(val_index, output_path, 'val')

    logging.info('Done!')


if __name__ == '__main__':
    init_logger(file_name='prepare_nyt.log')

    cmd_args = load_args()
    print_configs(cmd_args, print_function=logging.info)

    main(cmd_args)

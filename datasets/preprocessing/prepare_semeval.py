import argparse
import logging
import os
from typing import List, Dict, Any, Optional, Tuple

from datasets.preprocessing.common import load_relation_names, save_data, save_index
from zeshrex import PROJECT_PATH
from zeshrex.utils import init_logger, print_configs


def load_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser('Script for SemEval2010 Task 8 dataset preparation')

    parser.add_argument('--data_dir', type=str, default='./datasets/raw/SemEval2010_task8/')
    parser.add_argument('--train_file', type=str, default='TRAIN_FILE.TXT')
    parser.add_argument('--test_file', type=str, default='TEST_FILE_FULL.TXT')
    parser.add_argument('--relation_names_file', type=str, default='relation_names.txt')
    parser.add_argument('--output_dir', type=str, default='./datasets/prepared/SemEval2010_task8/')

    return parser.parse_args().__dict__


def load_data(
        data_path: os.PathLike,
        relation_names: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    logging.info(f'Loading dataset from {data_path}')

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
    indexes: List[int] = []
    relations: List[str] = []
    for chunk in raw_data_list:
        raw_id_and_text, raw_relation_type, raw_comment = chunk
        text_id = int(raw_id_and_text.split('\t')[0])
        text = raw_id_and_text.split('\t')[-1].strip('\n').strip('\"')
        relation_name = raw_relation_type.strip('\n')
        comment = raw_comment.strip('\n').split(':', 1)[-1].strip()
        if relation_names is not None:
            assert relation_name in relation_names, f'Unknown relation {relation_name}!'
        else:
            relations.append(relation_name)
        data.append({'index': text_id, 'relation': relation_name, 'text': text})
        indexes.append(text_id)

    if relation_names is None:
        relation_names = list(set(relations))

    logging.info(f'Data were collected including the following relations: {relation_names}')

    return data, indexes


def main(args: Dict[str, Any]) -> None:
    dataset_path = PROJECT_PATH / args['data_dir']
    output_path = PROJECT_PATH / args['output_dir']

    relation_names_file_path = dataset_path / args['relation_names_file']
    relation_names = load_relation_names(relation_names_file_path)

    train_data_path = dataset_path / args['train_file']
    test_data_path = dataset_path / args['test_file']

    train_data, train_index = load_data(train_data_path, relation_names)
    test_data, test_index = load_data(test_data_path, relation_names)

    joined_data = [*train_data, *test_data]

    save_data(joined_data, output_path)
    save_index(train_index, output_path, 'train')
    save_index(test_index, output_path, 'test')

    logging.info('Done!')


if __name__ == '__main__':
    init_logger(file_name='prepare_semeval.log')

    cmd_args = load_args()
    print_configs(cmd_args, print_function=logging.info)

    main(cmd_args)

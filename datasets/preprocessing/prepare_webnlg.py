import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from datasets.preprocessing.common import load_relation_names, save_data, save_index
from datasets.preprocessing.webnlg_corpus_reader import Benchmark, select_files
from zeshrex import PROJECT_PATH
from zeshrex.utils import init_logger, print_configs


def load_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser('Script for SemEval2010 Task 8 dataset preparation')

    parser.add_argument('--data_dir', type=str, default='./datasets/raw/WebNLG/')
    parser.add_argument('--train_data', type=str, default='./xml/train/')
    parser.add_argument('--test_data', type=str, default='./xml/test/')
    parser.add_argument('--dev_data', type=str, default='./xml/dev/')
    parser.add_argument('--relation_names_file', type=str, default='relation_names_top.txt')
    parser.add_argument('--relation_aliases_file', type=str, default='relation_aliases.json')
    parser.add_argument('--output_dir', type=str, default='./datasets/prepared/WebNLG/')

    return parser.parse_args().__dict__


def load_relation_aliases(file_path: os.PathLike) -> Dict[str, str]:
    assert Path(file_path).exists(), f'Relation aliases file not found! {file_path}'
    with open(file_path) as jf:
        alias_to_relation: Dict[str, str] = json.load(jf)
    return alias_to_relation


def load_data(
    data_path: os.PathLike,
    relation_names: Optional[List[str]] = None,
    relation_aliases: Optional[Dict[str, str]] = None,
    initial_index: int = 1,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    data_path = Path(data_path)

    logging.info(f'Loading dataset from {data_path}')
    assert data_path.exists(), f'Data file not found! {data_path}'

    if relation_names:
        logging.info(f'Loading data with the following relations: {relation_names}')
        logging.info('NOTE that relations not included in this list will be skipped!')
        relation_names_set = set(relation_names)
        assert len(relation_names_set) == len(relation_names), 'Duplicated relations found!'
    else:
        relation_names_set = set()

    # initialise Benchmark object
    b = Benchmark()

    # collect xml files
    files = select_files(str(data_path))

    # load files to Benchmark
    b.fill_benchmark(files)

    data: List[Dict[str, Any]] = []
    indexes: List[int] = []
    derived_relations: List[str] = []
    current_index = initial_index
    for entry in b.entries:
        for triplet in entry.modifiedtripleset.triples:
            sub_text, pred, obj_text = triplet.s, triplet.p, triplet.o

            sub_text = sub_text.replace('_', ' ').strip('\"').strip("\'")
            obj_text = obj_text.replace('_', ' ').strip('\"').strip("\'")

            relations = [relation for alias, relation in relation_aliases.items() if pred == alias]
            for single_relation in relations:
                if relation_names is not None and single_relation not in relation_names_set:
                    logging.debug(f'Skipping relation {single_relation} as it is not stated in relation names!')
                    continue
                derived_relations.append(single_relation)

                for lex in entry.lexs:
                    raw_text = lex.lex
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
                    text += first_beg_token + raw_text[first_beg_idx : (first_beg_idx + len(first_text))] + first_end_token
                    text += raw_text[(first_beg_idx + len(first_text)) : second_beg_idx]
                    text += second_beg_token + raw_text[second_beg_idx : (second_beg_idx + len(second_text))] + second_end_token
                    text += raw_text[(second_beg_idx + len(second_text)) :]

                    sample = {
                        'index': current_index,
                        'relation': single_relation,
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

    train_relation_names = random.sample(relation_names, int(len(relation_names) * 0.7))
    test_relation_names = [rel for rel in relation_names if rel not in train_relation_names]

    relation_aliases_file_path = dataset_path / args['relation_aliases_file']
    relation_aliases = load_relation_aliases(relation_aliases_file_path)

    train_data_path = dataset_path / args['train_data']
    test_data_path = dataset_path / args['test_data']
    dev_data_path = dataset_path / args['dev_data']

    train_data, train_index = load_data(train_data_path, relation_names, relation_aliases)
    test_data, test_index = load_data(
        test_data_path, relation_names, relation_aliases, initial_index=max(train_index) + 1
    )
    dev_data, dev_index = load_data(dev_data_path, relation_names, relation_aliases, initial_index=max(test_index) + 1)

    joined_data = [*train_data, *test_data, *dev_data]

    save_data(joined_data, output_path)
    save_index(train_index, output_path, 'train')
    save_index(test_index, output_path, 'test')
    save_index(dev_index, output_path, 'val')

    logging.info('Done!')


if __name__ == '__main__':
    init_logger(file_name='prepare_webnlg.log')

    cmd_args = load_args()
    print_configs(cmd_args, print_function=logging.info)

    main(cmd_args)

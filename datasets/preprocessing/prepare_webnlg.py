import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from datasets.preprocessing.common import load_relation_names, save_data, save_index
from datasets.preprocessing.webnlg_corpus_reader import Benchmark, select_files
from zeshrex import PROJECT_PATH
from zeshrex.utils import init_logger, print_configs


def load_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser('Script for SemEval2010 Task 8 dataset preparation')

    parser.add_argument('--data_dir', type=str, default='./datasets/raw/WebNLG2019_version21/')
    parser.add_argument('--train_data', type=str, default='./xml/train/')
    parser.add_argument('--test_data', type=str, default='./xml/test/')
    parser.add_argument('--dev_data', type=str, default='./xml/dev/')
    parser.add_argument('--relation_names_file', type=str, default='relation_names.txt')
    parser.add_argument('--relation_aliases_file', type=str, default='relation_aliases.json')
    parser.add_argument('--output_dir', type=str, default='./datasets/prepared/WebNLG2019_version21/')

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
    logging.info(f'Loading dataset from {data_path}')
    assert Path(data_path).exists(), f'Data file not found! {data_path}'

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
            sub, pred, obj = triplet.s, triplet.p, triplet.o

            sub = sub.replace('_', ' ').strip('\"').strip("\'")
            obj = obj.replace('_', ' ').strip('\"').strip("\'")

            relations = [relation for alias, relation in relation_aliases.items() if pred == alias]
            for single_relation in relations:
                if relation_names is not None and single_relation not in relation_names_set:
                    logging.debug(f'Skipping relation {single_relation} as it is not stated in relation names!')
                    continue
                derived_relations.append(single_relation)

                for lex in entry.lexs:
                    raw_text = lex.lex
                    try:
                        sub_begin = raw_text.index(sub)
                        obj_begin = raw_text.index(obj)
                    except ValueError:
                        continue
                    if sub_begin <= obj_begin:
                        first, second = sub, obj
                        first_beg, second_beg = sub_begin, obj_begin
                    else:
                        first, second = obj, sub
                        first_beg, second_beg = obj_begin, sub_begin

                    text = raw_text[:first_beg]
                    text += '<e1>' + raw_text[first_beg : (first_beg + len(first))] + '</e1>'
                    text += raw_text[(first_beg + len(first)) : second_beg]
                    text += '<e2>' + raw_text[second_beg : (second_beg + len(second))] + '</e2>'
                    text += raw_text[(second_beg + len(second)) :]
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

    relation_names_file_path = dataset_path / args['relation_names_file']
    relation_names = load_relation_names(relation_names_file_path)

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

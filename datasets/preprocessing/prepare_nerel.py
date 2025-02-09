import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
from tqdm import tqdm

from datasets.preprocessing.common import load_relation_names, save_data, save_index
from zeshrex import PROJECT_PATH
from zeshrex.utils import init_logger, print_configs


def load_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser('Script for SemEval2010 Task 8 dataset preparation')

    parser.add_argument('--data_dir', type=str, default='./datasets/raw/NEREL/')
    parser.add_argument('--train_data', type=str, default='./train/')
    parser.add_argument('--test_data', type=str, default='./test/')
    parser.add_argument('--dev_data', type=str, default='./dev/')
    parser.add_argument('--relation_names_file', type=str, default='./relation_names_top.tsv')
    parser.add_argument('--output_dir', type=str, default='./datasets/prepared/NEREL/')

    return parser.parse_args().__dict__


def split_sentences(text: str) -> Dict[Tuple[int, int], str]:
    sentences = nltk.sent_tokenize(text, language='russian')

    position_to_sentence_map: Dict[Tuple[int, int], str] = {}
    current_start = 0
    for sent in sentences:
        current_end = current_start + len(sent)
        position_to_sentence_map[(current_start, current_end)] = sent
        current_start = current_end + 1

    return position_to_sentence_map


def load_data(
    data_path: os.PathLike,
    relation_names: Optional[List[str]] = None,
    initial_index: int = 1,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    data_path = Path(data_path)
    logging.info(f'Loading dataset from {data_path}')
    assert data_path.exists(), f'Data file not found! {data_path}'

    if relation_names:
        logging.info(f'Loading data with the following relations: {list(relation_names.keys())}')
        logging.info('NOTE that relations not included in this list will be skipped!')
        relation_names_set = set(relation_names)
        assert len(relation_names_set) == len(relation_names), 'Duplicated relations found!'
    else:
        relation_names_set = set()

    data: List[Dict[str, Any]] = []
    indexes: List[int] = []
    derived_relations: List[str] = []
    current_index = initial_index

    entities_total_count = 0
    wrong_bound_entity_count = 0

    samples_total_count = 0
    processed_samples_count = 0
    missed_entities_sample_count = 0
    incorrect_insertion_sample_count = 0
    nested_entities_sample_count = 0

    document_level_sample_count = 0

    for file_name in tqdm(sorted(os.listdir(str(data_path)))):
        text_file_path = data_path / file_name
        if text_file_path.suffix != '.txt':
            continue
        annotation_file_path = text_file_path.parent / f'{text_file_path.stem}.ann'
        if not annotation_file_path.exists():
            logging.warning(f'Annotation file {annotation_file_path} not found! Skipping!')
            continue

        # -------------------------------------------
        # Raw text loading and splitting by sentences
        # -------------------------------------------
        with open(text_file_path, 'r') as txt_file:
            raw_text_elements = []
            raw_text_with_breaks_elements = []
            for line in txt_file:
                raw_text_with_breaks_elements.append(line)
                line_stripped = line.strip('\n')
                if len(line_stripped) == 0:
                    continue
                raw_text_elements.append(line_stripped)

        raw_text_with_breaks = ''.join(raw_text_with_breaks_elements)
        raw_text_sentences = split_sentences(' '.join(raw_text_elements))

        # ---------------------------
        # Annotation files processing
        # ---------------------------
        entities: Dict[str, Dict[str, str]] = {}
        with open(annotation_file_path, 'r') as ann_file:
            for line in ann_file:
                line = line.strip()
                tag = line.split('\t', 1)[0]

                # -----------------
                # Entity processing
                # -----------------
                if tag[0] == 'T':
                    _, entity_data, entity_text = line.split('\t')
                    try:
                        entity_type, entity_beg_idx, entity_end_idx = entity_data.split(' ')
                    except ValueError:
                        continue
                    entity_beg_idx, entity_end_idx = int(entity_beg_idx), int(entity_end_idx)
                    entities_total_count += 1

                    for sent_idx, ((sent_beg_idx, sent_end_idx), sent) in enumerate(raw_text_sentences.items()):
                        double_breaks_count = raw_text_with_breaks[:entity_end_idx].count('\n\n')
                        entity_beg_idx_shifted = entity_beg_idx - double_breaks_count
                        entity_end_idx_shifted = entity_end_idx - double_breaks_count

                        if entity_beg_idx_shifted >= sent_beg_idx and entity_end_idx_shifted <= sent_end_idx:
                            entity_beg_idx_inner = entity_beg_idx_shifted - sent_beg_idx
                            entity_end_idx_inner = entity_end_idx_shifted - sent_beg_idx

                            if sent[entity_beg_idx_inner:entity_end_idx_inner] != entity_text:
                                logging.debug('Wrong bounds of entity! Trying to correct indexes by text search...')
                                idx_correction = sent.find(entity_text)
                                if idx_correction != -1:
                                    logging.debug('Found entity text! Indexes location corrected!')
                                    entity_beg_idx_inner = idx_correction
                                    entity_end_idx_inner = entity_beg_idx_inner + len(entity_text)
                                else:
                                    logging.debug('Cannot correct indexes location, skipping...')
                                    wrong_bound_entity_count += 1
                                    break

                            entities[tag] = {
                                'entity_text': entity_text,
                                'sentence': sent,
                                'sentence_index': sent_idx,
                                'begin_idx': entity_beg_idx_inner,
                                'end_idx': entity_end_idx_inner,
                            }
                            break
                    else:
                        logging.debug('Entity not found in text!')
                        logging.debug(f'Entity: {entity_text}, text: {raw_text_with_breaks}')

                # ------------------------------------------
                # Relation processing (considering entities)
                # ------------------------------------------
                elif tag[0] == 'R':
                    samples_total_count += 1

                    _, relation_data = line.split('\t')
                    relation_type, first_arg, second_arg = relation_data.split(' ')
                    if relation_names is not None and relation_type not in relation_names_set:
                        logging.debug(f'Skipping relation {relation_type} as it is not stated in relation names!')
                        continue

                    processed_samples_count += 1
                    derived_relations.append(relation_type)

                    first_arg_tag, sub_entity_tag = first_arg.split(':')
                    second_arg_tag, obj_entity_tag = second_arg.split(':')
                    assert first_arg_tag == 'Arg1' and second_arg_tag == 'Arg2', 'Wrong tags of arguments!'

                    if sub_entity_tag not in entities or obj_entity_tag not in entities:
                        logging.debug('One of entities not found! Skipping...')
                        missed_entities_sample_count += 1
                        continue

                    sub_begin_idx = entities[sub_entity_tag]['begin_idx']
                    sub_end_idx = entities[sub_entity_tag]['end_idx']
                    obj_begin_idx = entities[obj_entity_tag]['begin_idx']
                    obj_end_idx = entities[obj_entity_tag]['end_idx']

                    # Single sentence - Sentence level
                    if entities[sub_entity_tag]['sentence'] == entities[obj_entity_tag]['sentence']:
                        target_sentence = entities[sub_entity_tag]['sentence']

                    # Two joined sentences - Document level
                    else:
                        document_level_sample_count += 1
                        if entities[sub_entity_tag]['sentence_index'] < entities[obj_entity_tag]['sentence_index']:
                            target_sentence = ' '.join(
                                [entities[sub_entity_tag]['sentence'], entities[obj_entity_tag]['sentence']]
                            )
                            # Update location of object entity to be relative in joined sentence
                            obj_entity_loc_shift = len(entities[sub_entity_tag]['sentence']) + 1
                            obj_begin_idx += obj_entity_loc_shift
                            obj_end_idx += obj_entity_loc_shift
                        else:
                            target_sentence = ' '.join(
                                [entities[obj_entity_tag]['sentence'], entities[sub_entity_tag]['sentence']]
                            )
                            # Update location of subject entity to be relative in joined sentence
                            sub_entity_loc_shift = len(entities[obj_entity_tag]['sentence']) + 1
                            sub_begin_idx += sub_entity_loc_shift
                            sub_end_idx += sub_entity_loc_shift

                    if sub_begin_idx <= obj_begin_idx:
                        first_beg_idx, second_beg_idx = sub_begin_idx, obj_begin_idx
                        first_end_idx, second_end_idx = sub_end_idx, obj_end_idx
                        first_beg_token, first_end_token = '<e1>', '</e1>'
                        second_beg_token, second_end_token = '<e2>', '</e2>'
                    else:
                        first_beg_idx, second_beg_idx = obj_begin_idx, sub_begin_idx
                        first_end_idx, second_end_idx = obj_end_idx, sub_end_idx
                        first_beg_token, first_end_token = '<e2>', '</e2>'
                        second_beg_token, second_end_token = '<e1>', '</e1>'

                    # First entity insertion
                    text = target_sentence[:first_beg_idx]
                    text += first_beg_token + target_sentence[first_beg_idx:first_end_idx] + first_end_token
                    text += target_sentence[first_end_idx:]

                    # Preparation for second entity insertion
                    target_sentence = text
                    if first_end_idx > second_beg_idx:
                        shift = 4
                        nested_entities_sample_count += 1
                    else:
                        shift = 9

                    # Second entity insertion
                    text = target_sentence[: second_beg_idx + shift]
                    text += (
                        second_beg_token
                        + target_sentence[second_beg_idx + shift : second_end_idx + shift]
                        + second_end_token
                    )
                    text += target_sentence[second_end_idx + shift :]

                    # Check if entities were inserted correctly
                    if (
                        text.find(first_beg_token) == -1
                        or text.find(first_end_token) == -1
                        or text.find(second_beg_token) == -1
                        or text.find(second_end_token) == -1
                    ):
                        logging.debug('Error in entities tagging during insersion! Skipping...')
                        incorrect_insertion_sample_count += 1
                        continue

                    sample = {
                        'index': current_index,
                        'relation': relation_type,
                        'text': text,
                    }
                    data.append(sample)
                    indexes.append(current_index)
                    current_index += 1

                else:
                    continue

    logging.info(
        'Entities with wrong bounds: {} out of {} ({:.2f}%)'.format(
            wrong_bound_entity_count,
            entities_total_count,
            wrong_bound_entity_count / entities_total_count * 100,
        )
    )
    logging.info(
        'Samples with missed entities: {} out of {} ({:.2f}%)'.format(
            missed_entities_sample_count,
            processed_samples_count,
            missed_entities_sample_count / processed_samples_count * 100,
        )
    )
    logging.info(
        'Samples with incorrectly inserted entities: {} out of {} ({:.2f}%)'.format(
            incorrect_insertion_sample_count,
            processed_samples_count,
            incorrect_insertion_sample_count / processed_samples_count * 100,
        )
    )
    logging.info(
        'Samples with nested entities: {} out of {} ({:.2f}%)'.format(
            nested_entities_sample_count,
            processed_samples_count,
            nested_entities_sample_count / processed_samples_count * 100,
        )
    )
    logging.info(
        'Document level samples: {} out of {} ({:.2f}%)'.format(
            document_level_sample_count,
            processed_samples_count,
            document_level_sample_count / processed_samples_count * 100,
        )
    )
    logging.info(
        'Skipped samples due to relation restriction: {} out of {} ({:.2f}%)'.format(
            samples_total_count - processed_samples_count,
            samples_total_count,
            (samples_total_count - processed_samples_count) / samples_total_count * 100,
        )
    )

    return data, indexes


def main(args: Dict[str, Any]) -> None:
    dataset_path = PROJECT_PATH / args['data_dir']
    output_path = PROJECT_PATH / args['output_dir']

    relation_names_file_path = (dataset_path / args['relation_names_file']) if args['relation_names_file'] else None
    relation_names = load_relation_names(relation_names_file_path) if relation_names_file_path else None

    train_data_path = dataset_path / args['train_data']
    test_data_path = dataset_path / args['test_data']
    dev_data_path = dataset_path / args['dev_data']

    # Download the required NLTK data if not already present
    nltk.download('punkt')
    nltk.download('punkt_tab')

    logging.info('Processing train data')
    train_data, train_index = load_data(train_data_path, relation_names)
    logging.info('------------------------------------------------------------------------------')

    logging.info('Processing test data')
    test_data, test_index = load_data(test_data_path, relation_names, initial_index=max(train_index) + 1)
    logging.info('------------------------------------------------------------------------------')

    logging.info('Processing val data')
    dev_data, dev_index = load_data(dev_data_path, relation_names, initial_index=max(test_index) + 1)
    logging.info('------------------------------------------------------------------------------')

    joined_data = [*train_data, *test_data, *dev_data]

    save_data(joined_data, output_path)
    save_index(train_index, output_path, 'train')
    save_index(test_index, output_path, 'test')
    save_index(dev_index, output_path, 'val')

    if relation_names_file_path:
        output_relation_names_path = output_path / f'relation_names{relation_names_file_path.suffix}'
        shutil.copy(relation_names_file_path, output_relation_names_path)

    logging.info('Done!')


if __name__ == '__main__':
    init_logger()

    cmd_args = load_args()
    print_configs(cmd_args, print_function=logging.info)

    main(cmd_args)

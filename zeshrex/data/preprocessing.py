from abc import abstractmethod
from typing import List, Optional, Any


class BasePreprocessor:
    @abstractmethod
    def __call__(self, text: str) -> Any:
        pass


class SentenceTokenizationPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, max_len: int, truncate_from_tail: bool = False):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.tokenizer.add_tokens(['\n'], special_tokens=True)
        self.pad_id = 0

        self.truncate_from_tail = truncate_from_tail

    def __call__(self, text):
        if self.truncate_from_tail:
            tokenized_dict = self.tokenizer.encode_plus(text, truncation=False)
            input_ids, input_mask = tokenized_dict['input_ids'], tokenized_dict['attention_mask']

            input_ids = input_ids[-self.max_len :]
            input_ids[0] = self.cls_id
            input_mask = input_mask[-self.max_len :]
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))
            input_mask += [0] * (self.max_len - len(input_mask))

            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len

        else:
            tokenized_dict = self.tokenizer.encode_plus(
                text, max_length=self.max_len, pad_to_max_length=True, truncation=True
            )
            input_ids, input_mask = tokenized_dict['input_ids'], tokenized_dict['attention_mask']

            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len

        return input_ids, input_mask


class RelationTokenizationPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, max_len: int, relation_tokens: Optional[List[str]] = None):
        self.tokenizer = tokenizer
        if relation_tokens is not None:
            self.tokenizer.add_special_tokens({'additional_special_tokens': relation_tokens})

        self.max_len = max_len

        self.sequence_a_segment_id = 0
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.cls_token_segment_id = 0  # TODO: is it equal to self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.mask_padding_with_zero = True  # TODO: move to params?

        self.pad_token = 0
        self.pad_token_segment_id = 0

    def __call__(self, text: str):
        tokens: List[str] = self.tokenizer.tokenize(text)

        e11_p = tokens.index('<e1>')  # the start position of entity1  # TODO: get these tokens from init args
        e12_p = tokens.index('</e1>')  # the end position of entity1
        e21_p = tokens.index('<e2>')  # the start position of entity2
        e22_p = tokens.index('</e2>')  # the end position of entity2

        # Replace the token
        tokens[e11_p] = '$'
        tokens[e12_p] = '$'
        tokens[e21_p] = '#'
        tokens[e22_p] = '#'

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        special_tokens_count = 1
        max_len = self.max_len - special_tokens_count

        assert e12_p < max_len and e22_p < max_len, 'Index of special token is grater than max length!'

        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        token_type_ids = [self.sequence_a_segment_id] * len(tokens)

        tokens = [self.cls_token] + tokens
        token_type_ids = [self.cls_token_segment_id] + token_type_ids
        # NOTE: we are filling all with cls_token_id as we have context only and no [SEP] token with 2nd sentence

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if self.mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_len - len(input_ids)
        input_ids = input_ids + ([self.pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if self.mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([self.pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        assert len(input_ids) == self.max_len, f'Error with input length {len(input_ids)} vs {self.max_len}'
        assert (
            len(attention_mask) == self.max_len
        ), f'Error with attention mask length {len(attention_mask)} vs {self.max_len}'
        assert (
            len(token_type_ids) == self.max_len
        ), f'Error with token type length {len(token_type_ids)} vs {self.max_len}'

        return input_ids, attention_mask, token_type_ids, e1_mask, e2_mask

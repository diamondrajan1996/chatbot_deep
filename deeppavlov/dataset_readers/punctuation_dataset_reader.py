import sys
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from string import punctuation as PUNCTUATION

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.dataset_readers.morphotagging_dataset_reader import read_infile as read_ud_infile


NO_PUNCT = "NONE"

def make_word_punct_sents(sent, max_punct_in_row=3):
    word_sent, word_indexes, punct_sent, last_punct = [], [], [], ""
    for i, word in enumerate(sent):
        if word in PUNCTUATION:
            if last_punct == NO_PUNCT:
                last_punct = word
            elif max_punct_in_row > 0 and len(last_punct) < 2 * max_punct_in_row-1:
                if last_punct.endswith(".") and all(x == "." for x in word):
                    last_punct += word
                else:
                    last_punct += " " + word
        else:
            if len(word_sent) > 0:
                punct_sent.append(last_punct)
            word_sent.append(word)
            word_indexes.append(i)
            last_punct = NO_PUNCT
    punct_sent.append(last_punct)
    return word_sent, punct_sent, word_indexes


def recalculate_head_indexes(heads, word_indexes):
    reverse_word_indexes = {index: i for i, index in enumerate(word_indexes)}
    answer = []
    for new_word_index, word_index in enumerate(word_indexes):
        head_index, new_head_index = word_index, None
        while new_head_index is None:
            head_index = heads[head_index] - 1
            if head_index >= 0:
                new_head_index = reverse_word_indexes.get(head_index)
            else:
                new_head_index = -1
        answer.append(new_head_index+1)    
    return answer


def read_ud_punctuation_file(infile, read_syntax=False, max_sents=-1):
    data = read_ud_infile(infile, read_syntax=read_syntax, max_sents=max_sents)
    word_sents, punct_sents = [], []
    for source_sent, tag_sent in data:
        word_sent, punct_sent, word_indexes = make_word_punct_sents(source_sent)
        word_sents.append(word_sent)
        if read_syntax:
            tag_sent, head_sent, dep_sent = tag_sent
            tag_sent = [tag_sent[i] for i in word_indexes]
            head_sent = recalculate_head_indexes(head_sent, word_indexes)
            dep_sent = [dep_sent[i] for i in word_indexes]
            punct_sent = (punct_sent, tag_sent, head_sent, dep_sent)
        punct_sents.append(punct_sent)
    return list(zip(word_sents, punct_sents))


def read_punctuation_file(infile, to_tokenize=False, max_sents=-1, 
                          min_length=5, max_length=60):
    lines = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line != "":
                if "..." in line:
                    continue
                lines.append(line)
                if max_sents != -1 and len(lines) >= max_sents:
                    break
    if to_tokenize:
        tokenizer = LazyTokenizer()
        sents = tokenizer(lines)
    else:
        sents = [sent.split() for sent in lines]
    answer = []
    for sent in sents:
        if len(sent) < min_length:
            continue
        if max_length is not None and len(sent) > max_length:
            continue

        word_sent, punct_sent, word_indexes = make_word_punct_sents(sent)
        answer.append((word_sent, punct_sent))
    return answer

@register('ud_punctuation_dataset_reader')
class UDPunctuationDatasetReader(DatasetReader):

    def read(self, data_path: Dict, data_types: Optional[List[str]] = None,
             data_formats=None, **kwargs) -> Dict[str, List]:
        """Reads UD dataset from data_path.

        Args:
            data_path: can be either
                1. a directory containing files. The file for data_type 'mode'
                is then data_path / {language}-ud-{mode}.conllu
                2. a list of files, containing the same number of items as data_types
            language: a language to detect filename when it is not given
            data_types: which dataset parts among 'train', 'dev', 'test' are returned

        Returns:
            a dictionary containing dataset fragments (see ``read_infile``) for given data types
        """
        if data_types is None:
            data_types = ["train", "valid", "test"]
        if data_formats is None:
            data_formats = ["ud"] * len(data_path)
        answer = {"train": [], "valid": [], "test": []}
        for data_type, infile, data_format in zip(data_types, data_path, data_formats):
            if data_type not in data_types:
                continue
            if data_format == "ud":
                answer[data_type] += read_ud_punctuation_file(infile, **kwargs)
            elif data_format == "tokenized":
                answer[data_type] += read_punctuation_file(infile, to_tokenize=False, **kwargs)
            elif data_format == "text":
                answer[data_type] += read_punctuation_file(infile, to_tokenize=True, **kwargs)
            else:
                continue
        return answer
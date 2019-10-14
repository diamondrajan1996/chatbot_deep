import sys
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from string import punctuation as PUNCTUATION

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.dataset_readers.morphotagging_dataset_reader import read_infile as read_ud_infile


NO_PUNCT = "NONE"

def make_word_punct_sents(sent):
    word_sent, punct_sent, last_punct = [], [], ""
    for i, word in enumerate(sent):
        if word in PUNCTUATION:
            if last_punct == NO_PUNCT:
                last_punct = word
            else:
                last_punct += " " + word
        else:
            if len(word_sent) > 0:
                punct_sent.append(last_punct)
            word_sent.append(word)
            last_punct = NO_PUNCT
    punct_sent.append(last_punct)
    return word_sent, punct_sent


def read_ud_punctuation_file(infile, max_sents=-1):
    data = read_ud_infile(infile, max_sents=max_sents)
    word_sents, punct_sents = [], []
    for source_sent, tag_sent in data:
        word_sent, punct_sent = make_word_punct_sents(source_sent)
        word_sents.append(word_sent)
        punct_sents.append(punct_sent)
    return list(zip(word_sents, punct_sents))

@register('ud_punctuation_dataset_reader')
class UDPunctuationDatasetReader(DatasetReader):

    def read(self, data_path: Dict, data_types: Optional[List[str]] = None,
             **kwargs) -> Dict[str, List]:
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
            data_types = ["train", "dev", "test"]
        answer = dict()
        for data_type, infile in zip(data_types, data_path):
            if data_type not in data_types:
                continue
            answer[data_type] = read_ud_punctuation_file(infile, **kwargs)
        return answer
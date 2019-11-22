import sys
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


def read_gapping_file(infile, indexes=None, skip_first_line=True, max_sents=-1):
    if indexes is None:
        indexes = [0, 3]
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        if skip_first_line:
            fin.readline()
        for line in fin:
            curr_answer = []
            line = line.strip()
            splitted = line.split("\t")
            if len(splitted) < 2:
                continue
            text, label = splitted[:2]
            label = int(label)
            if label == 1:
                for i, elem in enumerate(splitted[2:], 2):
                    if elem != "":
                        elem = [list(map(int, x.split(":"))) for x in elem.split()]
                    else:
                        elem = []
                    curr_answer.append(elem)
                curr_answer = [curr_answer[i] for i in indexes]
            else:
                curr_answer = [[] for _ in range(len(indexes))]
            answer.append((text, [label, curr_answer]))
            if max_sents != -1 and len(answer) >= max_sents:
                break
    return answer


@register("gapping_dataset_reader")
class GappingDatasetReader:

    def read(self, data_path: List[Union[Path, str]],
             data_types: Optional[List[str]] = None, **kwargs) -> Dict[str, List]:
        if data_types is None:
            data_types = ["train", "valid", "test"]
        if len(data_path) != len(data_types):
            raise ValueError("The number of files should be able to the number of data types.")
        data = {}
        for mode, filepath in zip(data_types, data_path):
            if mode == "valid":
                kwargs["max_sents"] = 100
            data[mode] = read_gapping_file(filepath, **kwargs)
        return data







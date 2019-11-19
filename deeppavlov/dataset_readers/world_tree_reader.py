# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from typing import List, Dict, Tuple, Union
import re

import numpy as np
import pandas as pd

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


np.random.seed(123)


@register('world_tree_reader')
class WorldTreeReader(DatasetReader):

    def read(self, data_path: str,
            train_type: str,
            submit: bool,
            full_test: bool) -> Dict[str, List[Tuple[List[str], List[Union[str, int]]]]]:

        data_path = expand_path(data_path)
        base_path = data_path / "annotation/expl-tablestore-export-2017-08-25-230344/tables"
        train_path = data_path / "questions/ARC-Elementary+EXPL-Train.tsv"
        valid_path = data_path / "questions/ARC-Elementary+EXPL-Dev.tsv"

        explanations_base = []

        for path, _, files in os.walk(base_path):
            for file in files:
                explanations_base += self.read_explanations(os.path.join(path, file))

        self.explanations_base = dict(explanations_base)

        train_data = self.build_train_data(train_path, train_type)
        valid_data = self.build_valid_data(valid_path)

        if submit:
            test_path = data_path / "questions/ARC-Elementary+EXPL-Test-Masked.tsv"
            test_data = self.build_submit_data(test_path)
        else:
            test_path = data_path / "questionsAndExplanations.tsv"
            if full_test:
                test_data = self.build_full_test_data(test_path)
            else:
                test_data = self.build_test_data(test_path)

        dataset = {"train": train_data, "valid": valid_data, "test": test_data}

        return dataset

    def read_explanations(self, path):
        header = []
        uid = None

        df = pd.read_csv(path, sep="\t")

        for name in df.columns:
            if name.startswith("[SKIP]"):
                if "UID" in name and not uid:
                    uid = name
            else:
                header.append(name)

        if not uid or len(df) == 0:
            warnings.warn("Possibly misformatted file: " + path)
            return []

        return df.apply(
            lambda r: (r[uid], " ".join(str(s) for s in list(r[header]) if not pd.isna(s)), ), 1).tolist()

    def build_train_data(self, train_path, train_type):
        df = pd.read_csv(train_path, sep="\t")
        df = df[["AnswerKey.1", "Question", "explanation"]]
        df = df.dropna()
        df["AQ"] = df.apply(
            lambda x: self.split_sample(x["Question"], x["AnswerKey.1"]), axis=1
        )
        df["AQ"] = df["AQ"].map(lambda x: x.lower())
        df["explanation"] = df["explanation"].map(self.id_to_text)
        if train_type == 'multi_hop':
            samples = list(df.apply(lambda x: self.multi_hop_preproc(x["AQ"], x["explanation"]), axis=1))
        else:
            if train_type != 'ranking':
                warnings.warn("Invalid training type. Ranking will be used instead")
            samples = list(df.apply(lambda x: self.ranking_preproc(x["AQ"], x["explanation"]), axis=1))
        samples = [el for sample in samples for el in sample]
        np.random.shuffle(samples)
        return samples

    def build_valid_data(self, valid_path):
        df = pd.read_csv(valid_path, sep="\t")
        df = df[["AnswerKey.1", "Question", "explanation"]]
        df = df.dropna()
        df["AQ"] = df.apply(lambda x: (self.split_sample(x["Question"], x["AnswerKey.1"]).lower(), ), axis=1)
        df["explanation"], df["label"] = zip(*df["explanation"].map(self.random_choice))
        samples = list(zip(df["AQ"] + df["explanation"], df["label"]))
        return samples

    def build_test_data(self, test_path):
        df = pd.read_csv(test_path, sep="\t", encoding="ISO-8859-1")
        df = df.loc[df["category"] == "Test"]
        df = df[["AnswerKey", "question", "explanation"]]
        df = df.dropna()
        df["AQ"] = df.apply(lambda x: (self.split_sample(x["question"], x["AnswerKey"]).lower(), ), axis=1)
        df["explanation"], df["label"] = zip(*df["explanation"].map(self.random_choice))
        samples = list(zip(df["AQ"] + df["explanation"], df["label"]))
        return samples

    def build_full_test_data(self, test_path):
        df = pd.read_csv(test_path, sep="\t", encoding="ISO-8859-1")
        df = df.loc[df["category"] == "Test"]
        df = df[["AnswerKey", "question", "explanation"]]
        df = df.dropna()
        df["AQ"] = df.apply(lambda x: (self.split_sample(x["question"], x["AnswerKey"]).lower(), ), axis=1)
        df["AQ"] = df["AQ"].map(lambda x: x + tuple(self.explanations_base.values()))
        df["label"] = df["explanation"].map(self.labels_build)
        samples = list(zip(df["AQ"], df["label"]))
        return samples

    def build_submit_data(self, test_path):
        df = pd.read_csv(test_path, sep="\t")
        df = df[["AnswerKey.1", "Question", "questionID"]]
        df = df.dropna()
        df["AQ"] = df.apply(lambda x: (self.split_sample(x["Question"], x["AnswerKey.1"]).lower(), ), axis=1)
        df["AQ"] = df["AQ"].map(lambda x: x + tuple(self.explanations_base.values()))
        df["label"] = df.apply(lambda x: (x["questionID"], ) + tuple(self.explanations_base.keys()), axis=1)
        samples = list(zip(df["AQ"], df["label"]))
        return samples

    def labels_build(self, explanation):
        explanation_ids = [exp.split("|")[0] for exp in explanation.split()]
        labels = [1 if el in explanation_ids else 0 for el in self.explanations_base.keys()]
        return labels

    def random_choice(self, explanation):
        explanation_ids = [exp.split("|")[0] for exp in explanation.split()]
        labels = [
            1 if el in explanation_ids else 0 for el in self.explanations_base.keys()
        ]
        zero_idx = [i for i, el in enumerate(labels) if el == 0]
        one_idx = [i for i, el in enumerate(labels) if el == 1]
        random_idx = np.random.choice(zero_idx, 500 - len(one_idx), replace=False).tolist()
        facts = [list(self.explanations_base.values())[i] for i in one_idx + random_idx]
        y = [1] * len(one_idx) + [0] * len(random_idx)
        return facts, y

    def id_to_text(self, explanation):
        explanation_ids = [exp.split("|")[0] for exp in explanation.split()]
        explanation_texts = [
            self.explanations_base[key]
            for key in explanation_ids
            if key in self.explanations_base.keys()
        ]
        return "__eot__".join(explanation_texts).strip()

    def split_sample(self, text, right_ans):
        abcd = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "F"}
        q = re.split(r"\(A\)", text)[0].strip()
        ans = re.split(r"\(" + right_ans + "\)", text)[1]
        ans = re.split(r"\(" + abcd[right_ans] + "\)", ans)[0].strip()
        return ans + ". " + q if ans[-1] != "." else ans + " " + q

    def multi_hop_preproc(self, question, explanation):
        explanation = explanation.split("__eot__")
        constructed_explanation = [""] + explanation[:-1]
        segment_a = [question + " " + el for el in constructed_explanation]
        random_facts = np.random.choice(list(self.explanations_base.values()), len(explanation), replace=False)
        positive_samples = [([seg, exp], 1) for seg, exp in zip(segment_a, explanation)]
        negative_samples = [([seg, fact], 0) for seg, fact in zip(segment_a, random_facts)]
        samples = positive_samples + negative_samples
        return samples

    def ranking_preproc(self, question, explanation):
        explanation = explanation.split("__eot__")
        random_facts = np.random.choice(list(self.explanations_base.values()), len(explanation), replace=False)
        positive_samples = [([question, exp], 1) for exp in explanation]
        negative_samples = [([question, exp], 0) for exp in random_facts]
        samples = positive_samples + negative_samples
        return samples
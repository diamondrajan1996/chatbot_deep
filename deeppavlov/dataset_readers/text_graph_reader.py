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

import csv
import re
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('text_graph_reader')
class TextGraphReader(DatasetReader):

    def read(self, data_path: str) -> Dict[str, List[Tuple[List[str], int]]]:

        data_path = expand_path(data_path)
        store_path = data_path / 'facts_store4.pickle'
        data_path = data_path / 'explanations3.tsv'
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            tsv_data = list(reader)
        data = [el for el in tsv_data[1:] if len(el[-1]) != 1]
        train_data, valid_data, test_data = [], [], []
        for el in data:
            if el[3] == 'Train':
                train_data.append(el)
            elif el[3] == 'Dev':
                valid_data.append(el)
            elif el[3] == 'Test':
                test_data.append(el)
        with open(store_path, 'rb') as f:
            fact_store = pickle.load(f)
        train_data = self.build_train_data(train_data, fact_store)
        test_data = self.build_submit_data('~/tg2019/questions/ARC-Elementary+EXPL-Test-Masked.tsv', fact_store)
        valid_data = self.build_test_data(valid_data, fact_store)
        dataset = {"train": train_data, "valid": valid_data, "test": test_data}
        return dataset

    def _build_train_data(self, data, store):
        texts, sample = [], []
        store = list(store.values())
        for el in data:
            #print(el)
            context = '. '.join(list(self.split_sample(el[0], el[1])))
            facts = el[-1].split('. ')
            #pos_sample = [([context, f], 1) for f in facts]
            #print(len(pos_sample))
            #print(pos_sample)
            #neg_facts = np.random.choice(store, len(pos_sample))
            #neg_sample = [([context, f], 0) for f in neg_facts]
            #print(len(neg_sample))
            #print(neg_sample)
            facts = [el+'.' if el[-1] != '.' else el for el in facts]
            pos_sample = [[context]+facts[:i+1] for i, f in enumerate(facts)]
            pos_sample = [([' '.join(el[:-1]), el[-1]], 1) for el in pos_sample]
            neg_facts = np.random.choice(store, len(pos_sample))
            neg_sample = [[context]+facts[:i]+[neg_facts[i]] for i, f in enumerate(neg_facts)]
            neg_sample = [([' '.join(el[:-1]), el[-1]], 0) for el in neg_sample]
            texts.extend(pos_sample)
            texts.extend(neg_sample)
        np.random.shuffle(texts)
        return texts

    def build_submit_data(self, path, store):
        df_q = pd.read_csv(path, sep='\t')
        que = df_q['Question']
        ans = df_q['AnswerKey']
        texts, sample = [], []
        full_ids = list(store.keys())
        facts = list(store.values())
        for q, a in zip(que, ans):
            # print(el)
            context = '. '.join(list(self.split_sample(q, a)))
            y = [1 for i, _ in enumerate(full_ids)]
            texts.append(([context] + facts, y))
        return texts


    def build_train_data(self, data, store):
        texts, sample = [], []
        store = list(store.values())
        for el in data:
            # print(el)
            context = '. '.join(list(self.split_sample(el[0], el[1])))
            facts = el[-1].split('. ')
            pos_sample = [([context, f], 1) for f in facts]
            neg_facts = np.random.choice(store, len(pos_sample))
            neg_sample = [([context, f], 0) for f in neg_facts]
            texts.extend(pos_sample)
            texts.extend(neg_sample)
        np.random.shuffle(texts)
        return texts

    def build_test_data(self, data, store):
        texts, sample = [], []
        full_ids = list(store.keys())
        facts = list(store.values())
        for el in data:
            #print(el)
            context = '. '.join(list(self.split_sample(el[0], el[1])))
            tmp_ids = [i.split('|')[0] for i in el[-2].split(" ")]
            y = [1 if i in tmp_ids else 0 for i in full_ids]
            one_y = [i for i, el in enumerate(y) if el == 1]
            zero_y = [i for i, el in enumerate(y) if el == 0]
            zero_y = np.random.choice(zero_y, 500-len(one_y))
            y_idx = one_y + zero_y.tolist()
            _facts = [facts[i] for i in y_idx]
            y = [1 for i in one_y] + [0 for i in zero_y]
            texts.append(([context]+_facts, y))
        np.random.shuffle(texts)
        return texts

    def _build_test_data(self, data, store):
        texts, sample = [], []
        full_ids = list(store.keys())
        facts = list(store.values())
        for el in data:
            # print(el)
            context = '. '.join(list(self.split_sample(el[0], el[1])))
            tmp_ids = [i.split('|')[0] for i in el[-2].split(" ")]
            y = [1 if i in tmp_ids else 0 for i in full_ids]
            texts.append(([context] + facts, y))
        np.random.shuffle(texts)
        return texts

    def split_sample(self, text, right_ans):
        abcd = {'A': 'B', 'B': 'C', 'C': 'D', 'D': 'E', 'E': 'F'}
        q = re.split(r'\(A\)', text)[0].strip()
        ans = re.split(r'\(' + right_ans + '\)', text)[1]
        ans = re.split(r'\(' + abcd[right_ans] + '\)', ans)[0].strip()
        ans = ans[0].title() + ans[1:]
        return ans, q

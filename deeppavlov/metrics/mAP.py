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


import numpy as np
import pandas as pd
import pickle

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.metrics_registry import register_metric


def average_precision(y_true, y_pred):
    order = np.flip(np.argsort(y_pred, -1), -1)
    y_true = [y_true[i] for i in order]
    precision = [sum(y_true[:i + 1]) / (i + 1) for i, el in enumerate(y_true) if el == 1]
    return sum(precision) / len(precision)


@register_metric('map')
def mean_average_precision(y_true, y_pred):
    precision = [average_precision(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)]
    return sum(precision) / len(precision)


@register_metric('submit_metric')
def submit_metric(y_true, y_pred):
    q_file = pd.read_csv('~/tg2019/questions/ARC-Elementary+EXPL-Test-Masked.tsv', sep='\t')
    data_path = '~/'
    data_path = expand_path(data_path)
    store_path = data_path / 'facts_store4.pickle'
    with open(store_path, 'rb') as f:
        fact_store = pickle.load(f)
    question_ids = list(q_file["questionID"])
    fact_ids = list(fact_store.keys())
    orders = np.flip(np.argsort(y_pred, -1), -1) 
    ordered_facts = [[fact_ids[i] for i in order] for order in orders]
    submit = [q +' ' + f for q, ids in zip(question_ids, ordered_facts) for f in ids]
    with open('prediction.txt', 'w+') as f:
        f.write('\n'.join(submit))
    return 1

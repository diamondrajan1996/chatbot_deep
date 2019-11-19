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

import warnings

import numpy as np
import math

from deeppavlov.core.common.metrics_registry import register_metric


def compute_ranks(true, pred):
    ranks = []

    #if not true or not pred:
    #    return ranks

    targets = list(true)

    targets = [i for i, el in enumerate(targets) if el == 1]
    pred = np.flip(np.argsort(pred, -1), -1)

    for i, pred_id in enumerate(pred):
        for true_id in targets:
            if pred_id == true_id:
                ranks.append(i + 1)
                targets.remove(pred_id)
                break

    # Example: Mercury_SC_416133
    if targets:
        warnings.warn('targets list should be empty, but it contains: ' + ', '.join(targets))

        for _ in targets:
            ranks.append(0)

    return ranks


def average_precision(ranks):
    total = 0.

    if not ranks:
        return total

    for i, rank in enumerate(ranks):
        precision = float(i + 1) / float(rank) if rank > 0 else math.inf
        total += precision

    return total / len(ranks)


@register_metric('map')
def mean_average_precision_score(gold, pred):
    total, count = 0., 0

    for y_t, y_p in zip(gold, pred):
        ranks = compute_ranks(y_t, y_p)
        score = average_precision(ranks)

        if not math.isfinite(score):
            score = 0.

        total += score
        count += 1

    mean_ap = total / count if count > 0 else 0.

    return mean_ap

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

from typing import List, Iterable, Union, Tuple, Dict
from logging import getLogger
import pprint
import time

import numpy as np

from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.core.common.registry import register
from deeppavlov.models.generative_chitchat.preprocessing import InputExample


logger = getLogger(__name__)


@register("test_net")
class TestNet(NNModel):
    def __init__(self, batch_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size

    def load(self, *args, **kwargs) -> None:
        pass

    def save(self, *args, **kwargs) -> None:
        pass

    def train_on_batch(self, samples_generator, y):
        logger.info("TRAIN_ON_BATCH")
        logger.info(f"samples_generator={pprint.pformat([sample.to_dict() for sample in samples_generator])}")
        logger.info(f"y={y}")
        time.sleep(5)
        return 1

    def __call__(self, samples_generator):
        logger.info("CALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
        logger.info(f"samples_generator={pprint.pformat([sample.to_dict() for sample in samples_generator])}")
        time.sleep(5)
        return [1 for i in samples_generator]

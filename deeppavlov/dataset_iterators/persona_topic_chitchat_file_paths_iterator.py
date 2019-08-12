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

from logging import getLogger
from pathlib import Path
from typing import Tuple, Iterator, Optional, Dict, List, Union
import json

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.data.utils import chunk_generator
from deeppavlov.models.generative_chitchat.preprocessing import InputExample

log = getLogger(__name__)


@register("persona_topic_chitchat_file_paths_iterator")
class PersonaTopicFilePathsIterator(DataLearningIterator):
    """Dataset iterator for json files datasets.
    It gets lists of file paths from the data dictionary and returns examples of data objects.

    Args:
        data: dict with keys ``'train'``, ``'valid'`` and ``'test'`` and values
        seed: random seed for data shuffling
        shuffle: whether to shuffle data during batching

    """

    def __init__(
        self, data: Dict[str, List[Union[str, Path]]], seed: Optional[int] = None, shuffle: bool = True, *args, **kwargs
    ) -> None:
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        super().__init__(data, seed, shuffle, *args, **kwargs)

    def _shard_generator(self, shards: List[Union[str, Path]], shuffle: bool = False) -> List[str]:
        shards_to_choose = list(shards)
        if shuffle:
            self.np_random.shuffle(shards_to_choose)
        for shard in shards_to_choose:
            log.info(f"Loaded shard from {shard}")
            dialogs = json.load(open(shard, encoding="utf-8"))
            examples = []
            for dialog_id, dialog in enumerate(dialogs):
                if len(dialog["dialog"]) < 2:
                    continue
                contexts = [dialog["dialog"][:i] for i in range(2, len(dialog["dialog"]) + 1)]
                sender_id2user = {usr["sender_id"]: usr for usr in dialog["users"]}
                examples.extend(
                    [
                        InputExample(
                            unique_id=f"{shard}:{dialog_id}:{context_id}",
                            profile=sender_id2user[context[-1]["sender_id"]]["profile"],
                            topic=sender_id2user[context[-1]["sender_id"]]["topics"],
                            context=[msg["text"] for msg in context[:-1]],
                            next_sentense=context[-1]["text"],
                        )
                        for context_id, context in enumerate(contexts)
                    ]
                )

            if shuffle:
                self.np_random.shuffle(examples)
            yield examples

    def gen_batches(
        self, batch_size: int, data_type: str = "train", shuffle: Optional[bool] = None
    ) -> Iterator[Tuple[str, str]]:
        if shuffle is None:
            shuffle = self.shuffle

        tgt_data = self.data[data_type]
        shard_generator = self._shard_generator(tgt_data, shuffle=shuffle)

        for shard in shard_generator:
            if not (batch_size):
                batch_size = len(shard)
            examples_generator = chunk_generator(shard, batch_size)
            for examples in examples_generator:
                yield (examples, [1 for i in examples])

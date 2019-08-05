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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('chitchat_bot_adapter')
class ChitChatBotAdapter(Component):
    """"""
    def __init__(self, **kwargs):
        pass

    @overrides
    def __call__(self, input_data, *args, **kwargs):
        if not isinstance(input_data, list):
            return 'Err, input_data must be th list type'
        if not input_data:
            return 'Good, input_data is empty list'
        if len(batch) > 0 and isinstance(batch[0], str):
            batch = [word_tokenize(utt) for utt in batchA]
        return batch

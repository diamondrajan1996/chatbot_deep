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

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path
from logging import getLogger

from bert_dp.preprocessing import convert_examples_to_features, InputExample
from bert_dp.tokenization import FullTokenizer

logger = getLogger(__name__)


@register('bert_preprocessor')
class BertPreprocessor(Component):
    # TODO: docs

    def __init__(self, vocab_file, do_lower_case=True, max_seq_length: int = 512, *args, **kwargs):
        self.max_seq_length = max_seq_length
        vocab_file = str(expand_path(vocab_file))
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def __call__(self, texts_a, texts_b=None):
        if texts_b is None:
            texts_b = [None] * len(texts_a)
        # TODO: add unique id
        examples = [InputExample(unique_id=0, text_a=text_a, text_b=text_b) for text_a, text_b in zip(texts_a, texts_b)]

        return convert_examples_to_features(examples, self.max_seq_length, self.tokenizer)


@register('bert_split_preprocessor')
class BertSplitPreprocessor(BertPreprocessor):
    # TODO: docs

    def __call__(self, texts):
        examples = []
        texts_a = texts[0]
        if len(texts) == 1:
            texts_b = [None] * len(texts[0])
            # TODO: add unique id
            ex = [InputExample(unique_id=0, text_a=text_a, text_b=text_b) for text_a, text_b in zip(texts_a, texts_b)]
            examples.append(ex)
        else:
            for t in texts[1:]:
                ex = [InputExample(unique_id=0, text_a=text_a, text_b=text_b) for text_a, text_b in
                      zip(texts_a, t)]
                examples.append(ex)

        return [convert_examples_to_features(el, self.max_seq_length, self.tokenizer) for el in examples]

class InputSplitFeatures(object):
  """A single set of features of data."""

  def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
    self.unique_id = unique_id
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
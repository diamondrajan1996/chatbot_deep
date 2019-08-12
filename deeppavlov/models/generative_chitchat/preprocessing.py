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
# %%


class InputExample(object):
    def __init__(self, unique_id, profile=None, topic=None, context=None, next_sentense=None):
        self.unique_id = unique_id
        self.profile = profile
        self.topic = topic
        self.context = context
        self.next_sentense = next_sentense

    def to_dict(self):
        return {
            "unique_id": self.unique_id,
            "profile": self.profile,
            "topic": self.topic,
            "context": self.context,
            "next_sentense": self.next_sentense,
        }


class SpecTokenTemplate(object):
    def __init__(self, start_segment=[], end_segment=[], start_sentense=[], end_sentense=[], between_sentense=[]):
        self.start_segment = start_segment
        self.end_segment = end_segment
        self.start_sentense = start_sentense
        self.end_sentense = end_sentense
        self.between_sentense = between_sentense


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def _add_spec_tokens(text_lines, spec_tokens):
    text_lines = [spec_tokens.start_sentense + text_line + spec_tokens.end_sentense for text_line in text_lines]
    text_lines = sum([[text_line, spec_tokens.between_sentense] for text_line in text_lines[:-1]] + [[text_lines[:-1]]])
    text_lines = spec_tokens.start_segment + text_lines + spec_tokens.end_segment
    return text_lines


# %%
def _truncate_seq(text_lines, spec_tokens, max_length, left_truncate=True):
    max_length -= len(spec_tokens.start_segment) + len(spec_tokens.end_segment) - len(spec_tokens.between_sentense)
    lines_lens = [
        len(text_line)
        + len(spec_tokens.start_sentense)
        + len(spec_tokens.end_sentense)
        + len(spec_tokens.between_sentense)
        for text_line in text_lines
    ]
    total_len = sum(lines_lens)
    if total_len > max_length:
        stop_index = len(lines_lens)
        length_rest = max_length
        lines_lens = list(reversed(lines_lens)) if left_truncate else lines_lens
        text_lines = list(reversed(text_lines)) if left_truncate else text_lines
        for i, lines_len in enumerate(lines_lens, 1):
            length_rest -= lines_len
            if length_rest < 0:
                stop_index = i
                break
        text_lines = text_lines[:stop_index]
        last_line = text_lines[-1]
        last_line = (
            list(reversed(list(reversed(last_line))[:length_rest])) if left_truncate else last_line[:length_rest]
        )
        text_lines[-1] = last_line
        text_lines = list(reversed(text_lines)) if left_truncate else text_lines
        # print(f'stop_index {stop_index}, length_rest {length_rest}, len {len(text_lines)}, last items {text_lines[0][:2]if left_truncate else text_lines[-1][-2:]}, sum {sum([len(i)for i in text_lines])}')
    # else:
    # print(f'len {len(text_lines)}, last items {text_lines[0][:2]if left_truncate else text_lines[-1][-2:]}, sum {sum([len(i)for i in text_lines])}')
    return text_lines


# # %%
# for i in range(0, 100):
#     # print(i)
#     _truncate_seq(
#         [list(range(0, 10)), list(range(10, 20)), list(range(20, 40))],
#         SpecTokenTemplate(
#             start_segment=[101], end_segment=[102], start_sentense=[103], end_sentense=[104], between_sentense=[105]
#         ),
#         i,
#     )
# %%


# %%


def convert_examples_to_features(
    examples,
    seq_length,
    tokenizer,
    profile_spec_tokens=SpecTokenTemplate(),
    topics_spec_tokens=SpecTokenTemplate(),
    context_spec_tokens=SpecTokenTemplate(),
    next_sentence_spec_tokens=SpecTokenTemplate(),
    profile_segment_idexes=[0, 1],
    topics_segment_idexes=[0, 1],
    context_segment_idexes=[0, 1],
    next_sentence_segment_idexes=[0, 1],
):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in examples:
        profile = None
        topic = None
        context = None
        next_sentense = None
        if example.profile:
            profile = [tokenizer.tokenize(text_line) for text_line in example.profile]
            profile = _truncate_seq(profile)
            profile = _add_spec_tokens(profile, profile_spec_tokens)


#     tokens_a = tokenizer.tokenize(example.text_a)

#     tokens_b = None
#     if example.text_b:
#       tokens_b = tokenizer.tokenize(example.text_b)

#     if tokens_b:
#       # Modifies `tokens_a` and `tokens_b` in place so that the total
#       # length is less than the specified length.
#       # Account for [CLS], [SEP], [SEP] with "- 3"
#       _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
#     else:
#       # Account for [CLS] and [SEP] with "- 2"
#       if len(tokens_a) > seq_length - 2:
#         tokens_a = tokens_a[0:(seq_length - 2)]

#     # The convention in BERT is:
#     # (a) For sequence pairs:
#     #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#     #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
#     # (b) For single sequences:
#     #  tokens:   [CLS] the dog is hairy . [SEP]
#     #  type_ids: 0     0   0   0  0     0 0
#     #
#     # Where "type_ids" are used to indicate whether this is the first
#     # sequence or the second sequence. The embedding vectors for `type=0` and
#     # `type=1` were learned during pre-training and are added to the wordpiece
#     # embedding vector (and position vector). This is not *strictly* necessary
#     # since the [SEP] token unambiguously separates the sequences, but it makes
#     # it easier for the model to learn the concept of sequences.
#     #
#     # For classification tasks, the first vector (corresponding to [CLS]) is
#     # used as as the "sentence vector". Note that this only makes sense because
#     # the entire model is fine-tuned.
#     tokens = []
#     input_type_ids = []
#     tokens.append("[CLS]")
#     input_type_ids.append(0)
#     for token in tokens_a:
#       tokens.append(token)
#       input_type_ids.append(0)
#     tokens.append("[SEP]")
#     input_type_ids.append(0)

#     if tokens_b:
#       for token in tokens_b:
#         tokens.append(token)
#         input_type_ids.append(1)
#       tokens.append("[SEP]")
#       input_type_ids.append(1)

#     input_ids = tokenizer.convert_tokens_to_ids(tokens)

#     # The mask has 1 for real tokens and 0 for padding tokens. Only real
#     # tokens are attended to.
#     input_mask = [1] * len(input_ids)

#     # Zero-pad up to the sequence length.
#     while len(input_ids) < seq_length:
#       input_ids.append(0)
#       input_mask.append(0)
#       input_type_ids.append(0)

#     assert len(input_ids) == seq_length
#     assert len(input_mask) == seq_length
#     assert len(input_type_ids) == seq_length

#     features.append(
#         InputFeatures(
#             unique_id=example.unique_id,
#             tokens=tokens,
#             input_ids=input_ids,
#             input_mask=input_mask,
#             input_type_ids=input_type_ids))
#   return features


# def _truncate_seq_pair(tokens_a, tokens_b, max_length):
#   """Truncates a sequence pair in place to the maximum length."""

#   # This is a simple heuristic which will always truncate the longer sequence
#   # one token at a time. This makes more sense than truncating an equal percent
#   # of tokens from each, since if one sequence is very short then each token
#   # that's truncated likely contains more information than a longer sequence.
#   while True:
#     total_length = len(tokens_a) + len(tokens_b)
#     if total_length <= max_length:
#       break
#     if len(tokens_a) > len(tokens_b):
#       tokens_a.pop()
#     else:
#       tokens_b.pop()
# %%
l = [[1, 2], [3, 4]]
list(reversed(l))
#%%

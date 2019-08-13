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


class InputExample(object):
    def __init__(self, unique_id, profile=None, topic=None, context=None, next_sentence=None):
        self.unique_id = unique_id
        self.profile = profile
        self.topic = topic
        self.context = context
        self.next_sentence = next_sentence

    def to_dict(self):
        return {
            "unique_id": self.unique_id,
            "profile": self.profile,
            "topic": self.topic,
            "context": self.context,
            "next_sentence": self.next_sentence,
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

    def to_dict(self):
        return {
            "unique_id": self.unique_id,
            "tokens": self.tokens,
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "input_type_ids": self.input_type_ids,
        }


def _add_spec_tokens(text_lines, spec_tokens_template):
    text_lines = [
        spec_tokens_template.start_sentense + text_line + spec_tokens_template.end_sentense
        for text_line in text_lines
        if text_line
    ]
    text_lines = [text_line + spec_tokens_template.between_sentense for text_line in text_lines[:-1]] + [text_lines[-1]]
    text_lines[0] = spec_tokens_template.start_segment + text_lines[0]
    text_lines[-1] = text_lines[-1] + spec_tokens_template.end_segment
    return text_lines


def _get_segment_indexes(text_lines, segment_idexes, counting_from_right_segment=False):
    segment_idexes = segment_idexes * (len(text_lines) // len(segment_idexes) + 1)
    text_lines = list(reversed(text_lines)) if counting_from_right_segment else text_lines
    indexes = [len(text_line) * [seg] for text_line, seg in zip(text_lines, segment_idexes)]
    indexes = list(reversed(indexes)) if counting_from_right_segment else indexes
    return indexes


def _truncate_seq(text_lines, spec_tokens_template, max_length, left_truncate=True):
    max_length -= (
        len(spec_tokens_template.start_segment)
        + len(spec_tokens_template.end_segment)
        - len(spec_tokens_template.between_sentense)
    )
    lines_lens = [
        len(text_line)
        + len(spec_tokens_template.start_sentense)
        + len(spec_tokens_template.end_sentense)
        + len(spec_tokens_template.between_sentense)
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
    return text_lines


def convert_one_example_to_feature(
    text_lines,
    unique_id,
    label,
    seq_length,
    tokenizer,
    spec_tokens_template,
    segment_idexes_template,
    left_truncate=True,
    counting_from_right_segment=True,
):
    text_lines = [tokenizer.tokenize(text_line) for text_line in text_lines]
    text_lines = _truncate_seq(text_lines, spec_tokens_template, seq_length, left_truncate=left_truncate)
    text_lines = _add_spec_tokens(text_lines, spec_tokens_template)
    input_type_ids = _get_segment_indexes(text_lines, segment_idexes_template, counting_from_right_segment)
    input_type_ids = sum(input_type_ids, [])
    tokens = sum(text_lines, [])
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    pad_num = seq_length - len(input_ids)
    features = InputFeatures(
        unique_id=f"{unique_id}:{label}",
        tokens=tokens,
        input_ids=input_ids + [0] * pad_num,
        input_mask=input_mask + [0] * pad_num,
        input_type_ids=input_type_ids + [0] * pad_num,
    )
    return features


def convert_examples_to_features(
    examples,
    seq_length,
    tokenizer,
    profile_spec_tokens_template=SpecTokenTemplate(),
    topic_spec_tokens_template=SpecTokenTemplate(),
    context_spec_tokens_template=SpecTokenTemplate(),
    next_sentence_spec_tokens_template=SpecTokenTemplate(),
    profile_segment_idexes_template=[0, 1],
    topic_segment_idexes_template=[0, 1],
    context_segment_idexes_template=[0, 1],
    next_sentence_segment_idexes_template=[0, 1],
):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in examples:
        contexts = []
        if example.profile:
            contexts.append(
                convert_one_example_to_feature(
                    example.profile,
                    example.unique_id,
                    "profile",
                    seq_length,
                    tokenizer,
                    profile_spec_tokens_template,
                    profile_segment_idexes_template,
                    left_truncate=True,
                    counting_from_right_segment=False,
                )
            )
        if example.topic:
            contexts.append(
                convert_one_example_to_feature(
                    example.topic,
                    example.unique_id,
                    "topic",
                    seq_length,
                    tokenizer,
                    topic_spec_tokens_template,
                    topic_segment_idexes_template,
                    left_truncate=True,
                    counting_from_right_segment=False,
                )
            )
        if example.context:
            contexts.append(
                convert_one_example_to_feature(
                    example.context,
                    example.unique_id,
                    "context",
                    seq_length,
                    tokenizer,
                    context_spec_tokens_template,
                    context_segment_idexes_template,
                    left_truncate=True,
                    counting_from_right_segment=True,
                )
            )
        if example.next_sentence:
            contexts.append(
                convert_one_example_to_feature(
                    [example.next_sentence],
                    example.unique_id,
                    "next_sentence",
                    seq_length,
                    tokenizer,
                    next_sentence_spec_tokens_template,
                    next_sentence_segment_idexes_template,
                    left_truncate=False,
                    counting_from_right_segment=False,
                )
            )
        features.append(contexts)
    return features

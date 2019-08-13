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

import pprint
import collections

from bert_dp.tokenization import FullTokenizer

from deeppavlov.models.generative_chitchat.preprocessing import (
    InputExample,
    convert_one_example_to_feature,
    SpecTokenTemplate,
)


def check_lens(res, max_length):
    for k, v in res.to_dict().items():
        if k in ["tokens", "unique_id"]:
            continue
        assert len(v) == max_length


def get_total_num_elements(data, sub_set):
    el2num = collections.Counter(data)
    return sum([el2num[el] for el in sub_set])


def test_convert_examples_to_features():
    tokenizer = FullTokenizer(
        "/home/denis/.deeppavlov/downloads/bert_models/rubert_cased_L-12_H-768_A-12_v1/vocab.txt", False
    )
    unique_id = 123
    data = [
        " ".join(list(map(str, range(10)))),
        " ".join(list(map(str, range(10, 20)))),
        " ".join(list(map(str, range(20, 30)))),
        " ".join(list(map(str, range(30, 40)))),
        " ".join(list(map(str, range(40, 50)))),
    ]
    tok2id = {
        "[PAD]": 0,
        "[unused1]": 1,
        "[unused2]": 2,
        "[unused3]": 3,
        "[unused4]": 4,
        "[unused5]": 5,
        "[unused6]": 6,
        "[unused7]": 7,
        "[unused8]": 8,
        "[unused9]": 9,
        "[unused10]": 10,
    }

    start_segment = ["[unused1]", "[unused2]"]
    start_sentense = ["[unused3]", "[unused4]"]
    end_sentense = ["[unused5]", "[unused6]"]
    between_sentense = ["[unused7]", "[unused8]"]
    end_segment = ["[unused9]", "[unused10]"]
    spec_tokens_template = SpecTokenTemplate(
        start_segment=start_segment,
        start_sentense=start_sentense,
        end_sentense=end_sentense,
        between_sentense=between_sentense,
        end_segment=end_segment,
    )
    segment_idexes_template = [0, 1, 2, 3]

    max_length = 10
    res = convert_one_example_to_feature(
        data,
        unique_id,
        max_length,
        max_length,
        tokenizer,
        spec_tokens_template=spec_tokens_template,
        segment_idexes_template=segment_idexes_template,
        left_truncate=True,
        counting_from_right_segment=True,
    )
    check_lens(res, max_length)
    assert get_total_num_elements(res.input_mask, [1]) == 10
    assert get_total_num_elements(res.input_type_ids, [0]) == 10
    for el in start_segment + start_sentense + end_sentense + end_segment:
        assert get_total_num_elements(res.input_ids, [tok2id[el]]) == 1
    assert get_total_num_elements(res.input_ids, [tok2id["[PAD]"]]) == 0

    max_length = 24
    res = convert_one_example_to_feature(
        data,
        unique_id,
        max_length,
        max_length,
        tokenizer,
        spec_tokens_template=spec_tokens_template,
        segment_idexes_template=segment_idexes_template,
        left_truncate=True,
        counting_from_right_segment=True,
    )
    check_lens(res, max_length)
    assert get_total_num_elements(res.input_mask, [0]) == 6 and get_total_num_elements(res.input_mask, [1]) == 18
    assert get_total_num_elements(res.input_type_ids, [0]) == 24
    for el in start_segment + start_sentense + end_sentense + end_segment:
        assert get_total_num_elements(res.input_ids, [tok2id[el]]) == 1
    assert get_total_num_elements(res.input_ids, [tok2id["[PAD]"]]) == 6

    max_length = 25
    res = convert_one_example_to_feature(
        data,
        unique_id,
        max_length,
        max_length,
        tokenizer,
        spec_tokens_template=spec_tokens_template,
        segment_idexes_template=segment_idexes_template,
        left_truncate=True,
        counting_from_right_segment=True,
    )
    check_lens(res, max_length)
    assert get_total_num_elements(res.input_mask, [1]) == 25
    assert (
        get_total_num_elements(res.input_type_ids, [0]) == 16 and get_total_num_elements(res.input_type_ids, [1]) == 9
    )
    for el in start_segment + end_segment + between_sentense:
        assert get_total_num_elements(res.input_ids, [tok2id[el]]) == 1
    for el in start_sentense + end_sentense:
        assert get_total_num_elements(res.input_ids, [tok2id[el]]) == 2
    assert get_total_num_elements(res.input_ids, [tok2id["[PAD]"]]) == 0

    max_length = 100
    res = convert_one_example_to_feature(
        data,
        unique_id,
        max_length,
        max_length,
        tokenizer,
        spec_tokens_template=spec_tokens_template,
        segment_idexes_template=segment_idexes_template,
        left_truncate=True,
        counting_from_right_segment=True,
    )
    check_lens(res, max_length)
    assert get_total_num_elements(res.input_mask, [0]) == 18 and get_total_num_elements(res.input_mask, [1]) == 82
    assert (
        get_total_num_elements(res.input_type_ids, [0]) == 2 + 16 * 2 + 18
        and get_total_num_elements(res.input_type_ids, [1]) == 16
        and get_total_num_elements(res.input_type_ids, [2]) == 16
        and get_total_num_elements(res.input_type_ids, [3]) == 16
    )
    for el in start_segment + end_segment:
        assert get_total_num_elements(res.input_ids, [tok2id[el]]) == 1
    for el in between_sentense:
        assert get_total_num_elements(res.input_ids, [tok2id[el]]) == 4
    for el in start_sentense + end_sentense:
        assert get_total_num_elements(res.input_ids, [tok2id[el]]) == 5
    assert get_total_num_elements(res.input_ids, [tok2id["[PAD]"]]) == 18
    assert str(res.input_type_ids) == str([0] * 18 + [3] * 16 + [2] * 16 + [1] * 16 + [0] * 16 + [0] * 18)
    assert len(res.tokens) == 82

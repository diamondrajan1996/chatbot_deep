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
from overrides import overrides
from typing import List, Dict, Union

import tensorflow as tf
from bert_dp.modeling import BertConfig, BertModel

from deeppavlov.core.common.registry import register
from deeppavlov.models.bert.bert_classifier import BertClassifierModel

logger = getLogger(__name__)


class BertFloatClassifierModel(BertClassifierModel):
    """Bert-based model for text classification.

    It uses output from [CLS] token and predicts labels using linear transformation.

    """
    @overrides
    def _init_graph(self):
        self._init_placeholders()

        self.bert = BertModel(config=self.bert_config,
                              is_training=self.is_train_ph,
                              input_ids=self.input_ids_ph,
                              input_mask=self.input_masks_ph,
                              token_type_ids=self.token_types_ph,
                              use_one_hot_embeddings=False,
                              )

        output_layer = self.bert.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [self.n_classes, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.n_classes], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            output_layer = tf.nn.dropout(output_layer, keep_prob=self.keep_prob_ph)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            if self.one_hot_labels:
                one_hot_labels = self.y_ph
            else:
                one_hot_labels = tf.one_hot(self.y_ph, depth=self.n_classes, dtype=tf.float32)

            self.y_predictions = tf.argmax(logits, axis=-1)
            if not self.multilabel:
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                self.y_probas = tf.nn.softmax(logits, axis=-1)
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                self.loss = tf.reduce_mean(per_example_loss)
            else:
                # we have multi-label case
                # some classes for some samples are true-labeled as `-1`
                # we should not take into account (loss) this values
                self.y_probas = tf.nn.sigmoid(logits)
                chosen_inds = tf.not_equal(one_hot_labels, -1)

                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)[chosen_inds])


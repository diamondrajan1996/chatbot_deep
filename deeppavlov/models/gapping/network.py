# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from typing import List, Any, Tuple, Union, Dict

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import xavier_initializer
import tensorflow.keras.backend as kb
from bert_dp.modeling import BertConfig, BertModel
from bert_dp.optimization import AdamWeightDecayOptimizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.models.bert.bert_sequence_tagger import BertSequenceNetwork, token_from_subtoken,\
    ExponentialMovingAverage
from deeppavlov.models.syntax_parser.network import gather_indexes, biaffine_attention
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.core.layers.tf_layers import bi_rnn
from deeppavlov.core.models.tf_model import LRScheduledTFModel

log = getLogger(__name__)


class GappingLoss:

    def __init__(self, positive_weight=1.0, best_only=False):
        self.positive_weight = positive_weight
        self.best_only = best_only
        self.__name__ = "gapping_loss"

    def __call__(self, y_true, y_pred, word_mask=None):
        if word_mask is None:
            word_mask = kb.ones_like(y_true, dtype="float")
        first_log = -kb.log(kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon()))
        second_log = -kb.log(kb.clip(1.0 - y_pred, kb.epsilon(), 1.0 - kb.epsilon()))
        # true_log = kb.max(y_true * first_log, axis=-1)
        y_true = kb.cast(y_true, dtype="float")
        first_loss = y_true * first_log * word_mask
        second_loss = (1.0 - y_true) * second_log * word_mask
        loss = self.positive_weight * kb.max(first_loss, axis=-1) + kb.max(second_loss, axis=-1)
        # if self.best_only:
        #     # p_wrong <= p_true => first_log_wrong >= first_log_true => diff_log = 0.0
        #     diff_log = kb.minimum(first_log - kb.expand_dims(true_log), 0.0)
        #     loss -= self.positive_weight * kb.min(diff_log, axis=-1)
        return loss, first_loss, second_loss

@register('bert_gapping_recognizer')
class BertGappingRecognizer(BertSequenceNetwork):

    def __init__(self,
                 keep_prob: float,
                 bert_config_file: str,
                 pretrained_bert: str = None,
                 attention_probs_keep_prob: float = None,
                 hidden_keep_prob: float = None,
                 embeddings_dropout: float = 0.0,
                 encoder_layer_ids: List[int] = (-1,),
                 encoder_dropout: float = 0.0,
                 optimizer: str = None,
                 weight_decay_rate: float = 1e-6,
                 state_size: int = 256,
                 use_infersent_similarity_layer: bool = False,
                 threshold: float = 0.5,
                 use_birnn: bool = False,
                 birnn_cell_type: str = 'lstm',
                 birnn_hidden_size: int = 256,
                 ema_decay: float = None,
                 ema_variables_on_cpu: bool = True,
                 positive_loss_weight: float = 1.0,
                 return_probas: bool = False,
                 freeze_embeddings: bool = False,
                 learning_rate: float = 1e-3,
                 bert_learning_rate: float = 2e-5,
                 min_learning_rate: float = 1e-07,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: float = 1.0,
                 **kwargs) -> None:
        self.embeddings_dropout = embeddings_dropout
        self.state_size = state_size
        self.use_infersent_similarity_layer = use_infersent_similarity_layer
        self.threshold = threshold
        self.use_birnn = use_birnn
        self.birnn_cell_type = birnn_cell_type
        self.birnn_hidden_size = birnn_hidden_size
        self.positive_loss_weight = positive_loss_weight
        self.loss_func = GappingLoss(self.positive_loss_weight)
        self.return_probas = return_probas
        super().__init__(keep_prob=keep_prob,
                         bert_config_file=bert_config_file,
                         pretrained_bert=pretrained_bert,
                         attention_probs_keep_prob=attention_probs_keep_prob,
                         hidden_keep_prob=hidden_keep_prob,
                         encoder_layer_ids=encoder_layer_ids,
                         encoder_dropout=encoder_dropout,
                         optimizer=optimizer,
                         weight_decay_rate=weight_decay_rate,
                         ema_decay=ema_decay,
                         ema_variables_on_cpu=ema_variables_on_cpu,
                         freeze_embeddings=freeze_embeddings,
                         learning_rate=learning_rate,
                         bert_learning_rate=bert_learning_rate,
                         min_learning_rate=min_learning_rate,
                         learning_rate_drop_div=learning_rate_drop_div,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         load_before_drop=load_before_drop,
                         clip_norm=clip_norm,
                         **kwargs)

    def _init_graph(self) -> None:
        self._init_placeholders()

        units = super()._init_graph()

        with tf.variable_scope('ner'):
            units = token_from_subtoken(units, self.y_masks_ph)
            if self.use_birnn:
                units, _ = bi_rnn(units,
                                  self.birnn_hidden_size,
                                  cell_type=self.birnn_cell_type,
                                  seq_lengths=self.seq_lengths,
                                  name='birnn')
                units = tf.concat(units, -1)
            units = tf.gather(units, self.sent_index_ph)
            # for verbs
            verb_states = gather_indexes(units, self.sent_verb_ph)
            # verb_states.shape = C * V * h
            verb_states = tf.layers.dense(verb_states, units=self.state_size)
            # gap_states.shape = C * L * h
            gap_states = tf.layers.dense(units, units=self.state_size)
            # similarities.shape = C * V * L
            # similarities = tf.keras.backend.batch_dot(verb_states, gap_states, axes=[1,2])
            similarities = biaffine_attention(kb.expand_dims(verb_states, axis=1), gap_states)[:,0]
            self.gap_probs = tf.nn.sigmoid(similarities)
        with tf.variable_scope("loss"):
            word_mask = tf.cast(self._get_tag_mask(), tf.float32) # B * L
            word_mask = tf.gather(word_mask, self.sent_index_ph) # C * L
            loss_tensor, first_loss_tensor, second_loss_tensor = self.loss_func(self.y_ph, self.gap_probs, word_mask)
            self.first_loss_tensor = first_loss_tensor
            self.second_loss_tensor = second_loss_tensor
            self.loss = tf.reduce_mean(loss_tensor)

    def _init_placeholders(self) -> None:
        super()._init_placeholders()
        self.y_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_ph')
        self.sent_index_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent_index_ph')
        self.sent_verb_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='sent_verb_ph')
        self.y_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_mask_ph')
        self.embeddings_keep_prob_ph = tf.placeholder_with_default(
            1.0, shape=[], name="embeddings_keep_prob_ph")

    def _build_feed_dict(self, input_ids, input_masks, y_masks, y_indexes, y=None):
        total_indexes_length = sum(len(x) for x in y_indexes)
        verb_indexes = np.zeros(shape=(total_indexes_length,), dtype=int)
        sent_indexes = np.zeros(shape=(total_indexes_length,), dtype=int)
        start = 0
        for i, curr_indexes in enumerate(y_indexes):
            end = start + len(curr_indexes)
            verb_indexes[start:end] = curr_indexes
            sent_indexes[start:end] = i
            start = end
        feed_dict = self._build_basic_feed_dict(input_ids, input_masks, train=(y is not None))
        feed_dict.update({self.sent_index_ph: sent_indexes,
                          self.sent_verb_ph: verb_indexes,
                          self.y_masks_ph: y_masks})
        if y is not None:
            y_all = np.concatenate(y, axis=0)
            feed_dict.update({self.y_ph: y_all})
        return feed_dict

    def __call__(self,
                 input_ids: Union[List[List[int]], np.ndarray],
                 input_masks: Union[List[List[int]], np.ndarray],
                 y_masks: Union[List[List[int]], np.ndarray],
                 y_verb_indexes) -> Union[List[List[int]], List[np.ndarray]]:
        """ Predicts tag indices for a given subword tokens batch

        Args:
            input_ids: indices of the subwords
            input_masks: mask that determines where to attend and where not to
            y_masks: mask which determines the first subword units in the the word

        Returns:
            Predictions indices or predicted probabilities fro each token (not subtoken)

        """
        feed_dict = self._build_feed_dict(input_ids, input_masks, y_masks, y_verb_indexes)
        if self.ema:
            self.sess.run(self.ema.switch_to_test_op)
        batch_verb_gap_probs, seq_lengths = self.sess.run([self.gap_probs, self.seq_lengths], feed_dict=feed_dict)
        # arreglamos las probilidades por las oraciones
        starts, ends = [0], []
        for elem in y_verb_indexes:
            end = starts[-1] + len(elem)
            starts.append(end)
            ends.append(end)
        starts.pop()
        sent_verb_gap_probs = [batch_verb_gap_probs[start:end] for start, end in zip(starts, ends)]
        # buscamos el verbo con la probabilidad de ellipsis mas grande
        batch_best_probs = [None] * len(sent_verb_gap_probs)
        verb_answer, gap_answer = [None] * len(sent_verb_gap_probs), [None] * len(sent_verb_gap_probs)
        probs_answer = []
        for i, curr_verb_gap_probs in enumerate(sent_verb_gap_probs):
            if len(curr_verb_gap_probs) > 0:
                best_verb_index = np.argmax(np.max(curr_verb_gap_probs, axis=1))
                best_verb_position = y_verb_indexes[i][best_verb_index]
                curr_best_probs = curr_verb_gap_probs[best_verb_index, :seq_lengths[i]]
                is_gap = (curr_best_probs >= self.threshold)
                is_gap[:best_verb_position+1] = False
                curr_gap_positions = list(np.where(is_gap)[0])
                gap_answer[i] = curr_gap_positions
                if len(curr_gap_positions) > 0:
                    verb_answer[i] = best_verb_position
                probs_answer.append(curr_best_probs)
            else:
                probs_answer.append(np.array([0.0] * seq_lengths[i]))
        if self.return_probas:
            return probs_answer
        else:
            answer = []
            for i, (curr_verb, curr_gaps) in enumerate(zip(verb_answer, gap_answer)):
                if curr_verb is None:
                    answer.append([[], []])
                else:
                    answer.append([[curr_verb], curr_gaps])
        return answer

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K


class QAFFN(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_size=768,
            dropout=0.1,
            residual=True,
            name='QAFFN'):
        """Feed-forward layers for question and answer.
        The input to this layer should be a two-elements tuple (q_embeddnig, a_embedding).
        The elements of tuple should be None or a tensor.

        In training, we should input both question embedding and answer embedding.

        In pre-inference, we should pass answer embedding only and save the embedding.

        In inference, we should pass the question embedding only and do a vector similarity search.

        Keyword Arguments:
            hidden_size {int} -- hidden size of feed-forward network (default: {768})
            dropout {float} -- dropout rate (default: {0.1})
            residual {bool} -- whether to use residual connection (default: {True})
            activation {[type]} -- activation function (default: {tf.keras.layers.ReLU()})
        """

        super(QAFFN, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual
        self.q_ffn = tf.keras.layers.Dense(
            units=hidden_size,
            use_bias=True
        )

        self.a_ffn = tf.keras.layers.Dense(
            units=hidden_size,
            use_bias=True
        )
        self.q_ffn.build([1, self.hidden_size])
        self.a_ffn.build([1, self.hidden_size])

    def _bert_to_ffn(self, bert_embedding, ffn_layer):
        if bert_embedding is not None:
            ffn_embedding = ffn_layer(bert_embedding)
            ffn_embedding = tf.keras.layers.ReLU()(ffn_embedding)
            if self.dropout > 0:
                ffn_embedding = tf.keras.layers.Dropout(
                    self.dropout)(ffn_embedding)

            if self.residual:
                ffn_embedding += bert_embedding
        else:
            ffn_embedding = None

        return ffn_embedding

    def call(self, inputs):
        q_bert_embedding, a_bert_embedding = inputs
        q_ffn_embedding = self._bert_to_ffn(q_bert_embedding, self.q_ffn)
        a_ffn_embedding = self._bert_to_ffn(a_bert_embedding, self.a_ffn)
        return q_ffn_embedding, a_ffn_embedding


def stack_two_tensor(tensor_a, tensor_b):
    if tensor_a is not None and tensor_b is not None:
        return tf.stack([tensor_a, tensor_b], axis=1)
    elif tensor_a is not None:
        return tf.expand_dims(tensor_a, axis=1)
    elif tensor_b is not None:
        return tf.expand_dims(tensor_b, axis=1)


class MedicalQAModel(tf.keras.Model):
    def __init__(self, name=''):
        super(MedicalQAModel, self).__init__(name=name)
        self.qa_ffn_layer = QAFFN()

    def call(self, inputs):
        q_bert_embedding = inputs.get('q_vectors')
        a_bert_embedding = inputs.get('a_vectors')
        q_embedding, a_embedding = self.qa_ffn_layer(
            (q_bert_embedding, a_bert_embedding))
        return stack_two_tensor(q_embedding, a_embedding)


class BioBert(tf.keras.Model):
    def __init__(self, name=''):
        super(BioBert, self).__init__(name=name)

    def _create_bert_input_tensor(self, inputs):
        with_question = None
        with_answer = None
        if 'q_input_ids' in inputs:
            with_question = True
            seq_length = tf.shape(inputs['q_input_ids'])[-1]

        if 'a_input_ids' in inputs:
            with_answer = True
            seq_length = tf.shape(inputs['a_input_ids'])[-1]

        # if with both q and a, convert them to (2*batch_size, seq_length)
        if with_question is not None and with_answer is not None:
            input_ids = tf.reshape(tf.stack(
                [inputs['q_input_ids'], inputs['a_input_ids']], axis=1), (-1, seq_length))
            input_masks = tf.reshape(tf.stack(
                [inputs['q_input_masks'], inputs['a_input_masks']], axis=1), (-1, seq_length))
            segment_ids = tf.reshape(tf.stack(
                [inputs['q_segment_ids'], inputs['a_segment_ids']], axis=1), (-1, seq_length))
        elif with_question is not None:
            input_ids = inputs['q_input_ids']
            input_masks = inputs['q_input_masks']
            segment_ids = inputs['q_segment_ids']
        elif with_answer is not None:
            input_ids = inputs['a_input_ids']
            input_masks = inputs['a_input_masks']
            segment_ids = inputs['a_segment_ids']
        else:
            raise ValueError('Inputs should contains either question or answer, got {0}'.format(
                list(inputs.keys())))
        return input_ids, input_masks, segment_ids, with_question, with_answer

    def _create_bert_output_tensor(self, bert_output, with_question, with_answer):
        a_bert_embedding = None
        q_bert_embedding = None
        max_seq_length = tf.shape(bert_output)[-2]
        hidden_size = tf.shape(bert_output)[-1]
        if with_question is not None and with_answer is not None:
            # reshape to (batch_size, 2, max_seq_len, hidden_size)
            # and split back to two tensors
            bert_output = tf.reshape(
                bert_output, (-1, 2, max_seq_length, hidden_size))
            q_bert_embedding, a_bert_embedding = tf.unstack(
                bert_output, axis=1)
        elif with_question is not None:
            q_bert_embedding = bert_output
        elif with_answer is not None:
            a_bert_embedding = bert_output

        return q_bert_embedding, a_bert_embedding

    def call(self, inputs):

        # inputs is dict with input features
        input_ids, input_masks, segment_ids, with_question, with_answer = self._create_bert_input_tensor(
            inputs)
        # pass to bert
        # with shape of (batch_size/2*batch_size, max_seq_len, hidden_size)
        # TODO(Alex): Add true bert model
        # Input: input_ids, input_masks, segment_ids all with shape (None, max_seq_len)
        # Output: a tensor with shape (None, max_seq_len, hidden_size)
        fake_bert_output = tf.expand_dims(tf.ones_like(
            input_ids, dtype=tf.float32), axis=-1)*tf.ones([1, 1, 768], dtype=tf.float32)

        q_bert_embedding, a_bert_embedding = self._create_bert_output_tensor(
            fake_bert_output, with_question, with_answer)
        return q_bert_embedding, a_bert_embedding


class MedicalQAModelwithBert(tf.keras.Model):
    def __init__(
            self,
            hidden_size=768,
            dropout=0.1,
            residual=True,
            activation=tf.keras.layers.ReLU(),
            name=''):
        super(MedicalQAModelwithBert, self).__init__(name=name)
        self.biobert = BioBert()
        self.qa_ffn_layer = QAFFN(
            hidden_size=hidden_size,
            dropout=dropout,
            residual=residual,
            activation=activation)

    def _avg_across_token(self, tensor):
        if tensor is not None:
            tensor = tf.reduce_mean(tensor, axis=1)
        return tensor

    def call(self, inputs):

        q_bert_embedding, a_bert_embedding = self.biobert(inputs)

        # according to USE, the DAN network average embedding across tokens
        q_bert_embedding = self._avg_across_token(q_bert_embedding)
        a_bert_embedding = self._avg_across_token(a_bert_embedding)

        q_embedding, a_embedding = self.qa_ffn_layer(
            (q_bert_embedding, a_bert_embedding))

        return stack_two_tensor(q_embedding, a_embedding)

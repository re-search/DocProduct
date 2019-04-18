from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import tensorflow as tf


class QAFFN(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_size=768,
            dropout=0.1,
            residual=True,
            activation=tf.keras.layers.ReLU(),
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
        self.activation = activation
        self.q_ffn = tf.keras.layers.Dense(
            units=hidden_size,
            use_bias=True,
            activation=activation
        )

        self.a_ffn = tf.keras.layers.Dense(
            units=hidden_size,
            use_bias=True,
            activation=activation
        )
        self.q_ffn.build([1, self.hidden_size])
        self.a_ffn.build([1, self.hidden_size])

    @tf.function
    def _bert_to_ffn(self, bert_embedding, ffn_layer):
        if bert_embedding is not None:
            ffn_embedding = ffn_layer(bert_embedding)
            if self.dropout > 0:
                ffn_embedding = tf.keras.layers.Dropout(
                    self.dropout)(ffn_embedding)

            if self.residual:
                try:
                    ffn_embedding += bert_embedding
                except:
                    raise ValueError('Incompatible shape for res connection, got {0}, {1}'.format(
                        ffn_embedding.shape, bert_embedding.shape))
        else:
            ffn_embedding = None

        return ffn_embedding

    def call(self, inputs):
        q_bert_embedding, a_bert_embedding = inputs
        q_ffn_embedding = self._bert_to_ffn(q_bert_embedding, self.q_ffn)
        a_ffn_embedding = self._bert_to_ffn(a_bert_embedding, self.a_ffn)
        return q_ffn_embedding, a_ffn_embedding


@tf.function
def qa_pair_loss(q_embedding, a_embedding):
    if q_embedding is not None and a_embedding is not None:
        q_embedding = q_embedding / \
            tf.norm(q_embedding, axis=-1, keepdims=True)
        a_embedding = a_embedding / \
            tf.norm(a_embedding, axis=-1, keepdims=True)
        similarity_vector = tf.reshape(
            tf.matmul(q_embedding, a_embedding, transpose_b=True), [-1, ])
        target = tf.reshape(tf.eye(q_embedding.shape[0]), [-1, ])
        loss = tf.keras.losses.binary_crossentropy(target, similarity_vector)
        return loss
    else:
        return 0


class MedicalQAModel(tf.keras.Model):
    def __init__(self, name=''):
        super(MedicalQAModel, self).__init__(name=name)
        self.qa_ffn_layer = QAFFN()

    def call(self, inputs):
        q_bert_embedding = inputs.get('q_vectors')
        a_bert_embedding = inputs.get('a_vectors')

        return self.qa_ffn_layer((q_bert_embedding, a_bert_embedding))


class BioBert(tf.keras.Model):
    def __init__(self, name=''):
        super(BioBert, self).__init__(name=name)

    @tf.function
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
        if with_question and with_answer:
            input_ids = tf.reshape(tf.stack(
                [inputs['q_input_ids'], inputs['a_input_ids']], axis=1), (-1, seq_length))
            input_masks = tf.reshape(tf.stack(
                [inputs['q_input_masks'], inputs['a_input_masks']], axis=1), (-1, seq_length))
            segment_ids = tf.reshape(tf.stack(
                [inputs['q_segment_ids'], inputs['a_segment_ids']], axis=1), (-1, seq_length))
        elif with_question:
            input_ids = inputs['q_input_ids']
            input_masks = inputs['q_input_masks']
            segment_ids = inputs['q_segment_ids']
        elif with_answer:
            input_ids = inputs['a_input_ids']
            input_masks = inputs['a_input_masks']
            segment_ids = inputs['a_segment_ids']
        else:
            raise ValueError('Inputs should contains either question or answer, got {0}'.format(
                list(inputs.keys())))
        return input_ids, input_masks, segment_ids, with_question, with_answer

    @tf.function
    def _create_bert_output_tensor(self, bert_output, with_question, with_answer):
        max_seq_length = tf.shape(bert_output)[-2]
        hidden_size = tf.shape(bert_output)[-1]
        true_tensor = tf.convert_to_tensor(True)
        if with_question is not None and with_answer is not None:
            # reshape to (batch_size, 2, max_seq_len, hidden_size)
            # and split back to two tensors
            bert_output = tf.reshape(
                bert_output, (-1, 2, max_seq_length, hidden_size))
            q_bert_embedding, a_bert_embedding = tf.unstack(
                bert_output, axis=1)
        elif with_question is not None:
            q_bert_embedding = bert_output
            a_bert_embedding = None
        elif with_answer is not None:
            a_bert_embedding = bert_output
            q_bert_embedding = None
        else:
            a_bert_embedding = None
            q_bert_embedding = None
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

    @tf.function
    def _avg_across_token(self, tensor):
        if tensor is not None:
            tensor = tf.reduce_mean(tensor, axis=1)
        return tensor

    def call(self, inputs):
        q_bert_embedding, a_bert_embedding = self.biobert(inputs)

        # according to USE, the DAN network average embedding across tokens
        q_bert_embedding = self._avg_across_token(q_bert_embedding)
        a_bert_embedding = self._avg_across_token(a_bert_embedding)

        return self.qa_ffn_layer((q_bert_embedding, a_bert_embedding))

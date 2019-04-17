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

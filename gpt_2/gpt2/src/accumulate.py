import argparse
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
import time


class AccumulatingOptimizer(object):
    def __init__(self, opt, var_list):
        self.opt = opt
        self.var_list = var_list
        self.aggregation = tf.VariableAggregation.SUM
        self.accum_vars = {tv: tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False, aggregation=self.aggregation, name='accum_var{0}'.format(tv_ind))
                           for tv_ind, tv in enumerate(var_list)}
        self.total_loss = tf.Variable(
            tf.zeros(shape=[], dtype=tf.float32), aggregation=self.aggregation, name='total_loss')
        self.count_loss = tf.Variable(
            tf.zeros(shape=[], dtype=tf.float32), aggregation=self.aggregation, name='count_loss')

    def reset(self):
        updates = [tv.assign(tf.zeros_like(tv))
                   for tv in self.accum_vars.values()]
        updates.append(self.total_loss.assign(
            tf.zeros(shape=[], dtype=tf.float32)))
        updates.append(self.count_loss.assign(
            tf.zeros(shape=[], dtype=tf.float32)))
        with tf.control_dependencies(updates):
            return 0.0

    def compute_gradients(self, loss):
        grads = self.opt.compute_gradients(
            loss, self.var_list, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        updates = [self.accum_vars[v].assign_add(g) for (g, v) in grads]
        updates.append(self.total_loss.assign_add(loss))
        updates.append(self.count_loss.assign_add(
            tf.constant(1.0)))
        with tf.control_dependencies(updates):
            return 0.0

    def apply_gradients(self):
        grads = [(g, v) for (v, g) in self.accum_vars.items()]
        with tf.control_dependencies([self.opt.apply_gradients(grads)]):
            return self.total_loss / self.count_loss

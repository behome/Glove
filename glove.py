#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 20:22
# @Author  : 宋继贤
# @Site    : 
# @File    : glove.py
# @Software: PyCharm


import os
from collections import Counter, defaultdict
from random import shuffle
import tensorflow as tf


from Hparams import hparams


class GloveModel:

    def __init__(self, hparams):
        self._hparams = hparams
        self.focal_input = tf.placeholder(tf.int32, shape=[self._hparams.batch_size], name='focal_words')
        self.context_input = tf.placeholder(tf.int32, shape=[self._hparams.batch_size], name='context_word')
        self.cooccurrence_count = tf.placeholder(tf.float32, shape=[self._hparams.batch_size], name='cooccurrence_count')

    def generate_cooccurrence_matrix(self):
        pass

    def build(self):
        count_max = tf.constant(self._hparams.cooccrrence_cap, dtype=tf.float32, name='count_max')
        scaling_factor = tf.constant(self._hparams.scaling_factor, dtype=tf.float32, name='scaling_factor')
        focal_embedding = tf.get_variable("focal_embedding",
                                          shape=[self._hparams.vocab_size, self._hparams.embedding_size],
                                          initializer=tf.random_uniform_initializer(-1.0, 1.0, 123))
        context_embedding = tf.get_variable("context_embedding",
                                            shape=[self._hparams.vocab_size, self._hparams.embedding_size],
                                            initializer=tf.random_uniform_initializer(-1.0, 1.0, 123))
        focal_emb = tf.nn.embedding_lookup(focal_embedding, self.focal_input)
        context_emb = tf.nn.embedding_lookup(context_embedding, self.context_input)
        focal_bias = tf.get_variable("focal_bias",
                                     shape=[self._hparams.vocab_size],
                                     initializer=tf.random_uniform_initializer(-1.0, 1.0, 123))
        context_bias = tf.get_variable("context_bias",
                                       shape=[self._hparams.vocab_size],
                                       initializer=tf.random_uniform_initializer(-1.0, 1.0, 123))
        f_b = tf.nn.embedding_lookup(focal_bias, self.focal_input)
        c_b = tf.nn.embedding_lookup(context_bias, self.context_input)
        weight_factor = tf.minimum(1.0, tf.pow(tf.div(self.cooccurrence_count, count_max), scaling_factor))
        embedding_product = tf.reduce_sum(tf.multiply(focal_emb, context_emb), 1)
        log_cooccurrences = tf.log(self.cooccurrence_count)

        distance_expr = tf.square(tf.add_n([
            embedding_product, f_b, c_b, tf.negative(log_cooccurrences)
        ]))
        self.loss = tf.reduce_sum(tf.multiply(weight_factor, distance_expr))
        self.final_embedding = tf.add(focal_embedding, context_embedding)

    def train(self):
        pass


def _context_window(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = region[max(start_index, 0):min(i-1, len(region))+1]
        right_context = region[max(i+1, 0):min(end_index, len(region))+1]
        yield left_context, word, right_context


if __name__ == '__main__':
    model = GloveModel(hparams)
    region = 'abcdefg'
    fc = _context_window(region, 2, 2)
    for i in range(len(region)-1):
        fc.__next__()
    left_context, word, right_context = fc.__next__()
    print(left_context, word, right_context)

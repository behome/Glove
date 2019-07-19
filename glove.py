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
import codecs


from Hparams import hparams


class GloveModel:

    def __init__(self, hparams):
        self._hparams = hparams
        self.focal_input = tf.placeholder(tf.int32, shape=[self._hparams.batch_size], name='focal_words')
        self.context_input = tf.placeholder(tf.int32, shape=[self._hparams.batch_size], name='context_word')
        self.cooccurrence_count = tf.placeholder(tf.float32, shape=[self._hparams.batch_size], name='cooccurrence_count')

    def generate_cooccurrence_matrix(self, corpus):
        word_count = Counter()
        coocurrence_counts = defaultdict(float)
        for region in corpus:
            word_count.update(region)
            for left_context, word, right_context in _context_window(region, self._hparams.left_context_size, self._hparams.right_context_size):
                for i, context_word in enumerate(left_context):
                    coocurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(right_context):
                    coocurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(coocurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus")
        self.words = [word for word, count in word_count.most_common(self._hparams.vocab_size)
                      if count >= self._hparams.min_occurrences]
        self.word_to_id = {word: i for i, word in enumerate(self.words)}
        self.cooccurrence_matrix = {
            (self.word_to_id[words[0]], self.word_to_id[words[1]]):count
            for words, count in coocurrence_counts.items()
            if words[0] in self.word_to_id and words[1] in self.word_to_id
        }

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
        batches = self.get_batches()
        opt = tf.train.GradientDescentOptimizer(self._hparams.learning_rate).minimize(self.loss)

        total_steps = 0
        with tf.Session() as sess:
            tf.global_variables_initializer.run()
            for epoch in range(self._hparams.num_epochs):
                shuffle(batches)
                for batch_index, batch in enumerate(batches):
                    i_s, j_s, counts = batch
                    if len(counts) != self._hparams.batch_size:
                        continue
                    feed_dict = {
                        self.focal_input: i_s,
                        self.context_input: j_s,
                        self.cooccurrence_count: counts
                    }
                    sess.run(opt, feed_dict=feed_dict)
                    total_steps += 1
                    if total_steps % 100 == 0:
                        loss = sess.run(self.loss, feed_dict=feed_dict)
                        print("The loss in %d step is %0.3f" %(total_steps, loss))
            self.embeddings = self.final_embedding.eval()

    def get_batches(self):
        if self.cooccurrence_matrix is None:
            raise ValueError("The co-occurrence matrix have not been created")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self._hparams.batch_size, i_indices, j_indices, counts))
        pass


def _context_window(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = region[max(start_index, 0):min(i-1, len(region))+1]
        right_context = region[max(i+1, 0):min(end_index, len(region))+1]
        yield left_context, word, right_context


def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield  tuple(sequence[i:i+batch_size] for sequence in sequences)


if __name__ == '__main__':
    model = GloveModel(hparams)
    corpus = []
    with codecs.open('./data.txt', 'r', 'utf-8') as fin:
        for line in fin.readlines():
            corpus.append(line.strip())

    model.generate_cooccurrence_matrix(corpus)
    print(model.word_to_id)
    print(model.cooccurrence_matrix)

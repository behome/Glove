#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 20:27
# @Author  : 宋继贤
# @Site    : 
# @File    : Hparams.py
# @Software: PyCharm

import tensorflow as tf

hparams = tf.contrib.training.HParams(
    embedding_size=300,
    vocab_size=7000,
    left_context_size=3,
    right_context_size=3,
    batch_size=512,
    scaling_factor=0.75,
    learning_rate=0.05,
    min_occurrences=1,
    cooccrrence_cap=100
)
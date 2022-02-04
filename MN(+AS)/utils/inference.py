# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
import thumt.utils.common as utils

from collections import namedtuple
from tensorflow.python.util import nest


def create_predict_graph(models, features, params):
    if not isinstance(models, (list, tuple)):
        raise ValueError("'models' must be a list or tuple")

    features = copy.copy(features)
    model_fns = [model.get_predict_func() for model in models]

    scores, alpha = model_fns[0](features)

    return features["text"][:, :], tf.expand_dims(tf.argmax(scores, axis=1), -1), scores, alpha[:,:]

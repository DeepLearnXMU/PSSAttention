# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers
import numpy as np

def model_graph(features, mode, params):
    vocab_size = len(params.vocabulary)

    with tf.variable_scope("embedding"):
        if mode == "predict":
            emb_A = tf.get_variable("embedding_A", [vocab_size, params.embedding_size])
            emb_B = tf.get_variable("embedding_B", [vocab_size, params.embedding_size])
            emb_C = tf.get_variable("embedding_C", [vocab_size, params.embedding_size])
        elif mode == "train":
            emb_A = tf.get_variable("embedding_A",
                                    [vocab_size, params.embedding_size],
                                    initializer=tf.constant_initializer(np.loadtxt(params.pretrained_embedding)))
            emb_B = tf.get_variable("embedding_B",
                                    [vocab_size, params.embedding_size],
                                    initializer=tf.constant_initializer(np.loadtxt(params.pretrained_embedding)))
            emb_C = tf.get_variable("embedding_C",
                                    [vocab_size, params.embedding_size],
                                    initializer=tf.constant_initializer(np.loadtxt(params.pretrained_embedding)))

    with tf.variable_scope("model_param"):
        M = tf.get_variable("M", [params.embedding_size, params.embedding_size])
        W = tf.get_variable("W", [params.embedding_size, 3])
        H = tf.get_variable("H", [params.embedding_size, params.embedding_size])

    text_inputs_A = tf.nn.embedding_lookup(emb_A, features["text"])
    aspect_inputs_B = tf.nn.embedding_lookup(emb_B, features["aspect"])
    text_inputs_C = tf.nn.embedding_lookup(emb_C, features["text"])

    aspect_Vector = tf.reduce_mean(aspect_inputs_B, 1)
    mem_mask = tf.sequence_mask(features["text_length"],
                                maxlen=tf.shape(text_inputs_A)[1],
                                dtype=tf.float32)
    ret = (1.0 - mem_mask) * -1e9

    for i in range(params.hops):
        alpha = tf.nn.softmax(ret + tf.reduce_sum(tf.multiply(text_inputs_A, tf.expand_dims(tf.matmul(aspect_Vector, M), 1)), 2))
        o = tf.reduce_sum(tf.multiply(text_inputs_C, tf.expand_dims(alpha, 2)), 1)
        aspect_Vector = o + tf.matmul(aspect_Vector, H)

    predict_socres = tf.matmul(aspect_Vector, W)

    if mode == "predict":
        return tf.nn.softmax(predict_socres), alpha
    
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
            logits=predict_socres,
            labels=features["polarity"],
            smoothing=params.label_smoothing,
            normalize=True
        )
        
    class_loss =  tf.reduce_mean(ce)

    attention_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.multiply(features["attention_mask"], alpha) - features["attention_value"]), axis = 1))
    
    return class_loss, attention_loss


class FINAL_BL_MN(interface.SEMEVALModel):

    def __init__(self, params, scope="FINAL_BL_MN"):
        super(FINAL_BL_MN, self).__init__(params=params, scope=scope)


    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                class_loss, attention_loss = model_graph(features, "train", params)
                return class_loss, attention_loss

        return training_fn

    def get_predict_func(self):
        def predict_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                score, alpha = model_graph(features, "predict", params)
            return score, alpha

        return predict_fn

    @staticmethod
    def get_name():
        return "FINAL_BL_MN"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            unk="<unk>",
            eos="<eos>",
            bos="<eos>",
            embedding_size=300,
            hidden_size=300,
            append_eos=False,
            dropout=0.1,
            use_variational_dropout=False,
            label_smoothing=0.0,
            constant_batch_size=True,
            batch_size=128,
            max_length=60,
            clip_grad_norm=50.0
        )

        return params

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SEMEVALModel(object):

    def __init__(self, params, scope):
        self._scope = scope
        self._params = params

    def get_training_func(self, initializer, regularizer=None):
        raise NotImplementedError("Not implemented")
        
    def get_predict_func(self):
        raise NotImplementedError("Not implemented")

    
    @staticmethod
    def get_name():
        raise NotImplementedError("Not implemented")

    @staticmethod
    def get_parameters():
        raise NotImplementedError("Not implemented")

    @property
    def parameters(self):
        return self._params

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core

import numpy as np


class ParameterTags(object):
    BIAS = 'BIAS'
    WEIGHT = 'WEIGHT'
    COMPUTED_PARAM = 'COMPUTED_PARAM'


class ParameterType(object):
    DENSE = 'dense'
    SPARSE = 'sparse'


class ParameterInfo(object):

    def __init__(
            self, param_id, param, key=None, shape=None, length=None,
            grad=None, blob_copy=None):
        assert isinstance(param, core.BlobReference)
        self.param_id = param_id
        self.name = str(param)
        self.blob = param
        self.key = key
        self.shape = shape
        self.size = None if shape is None else np.prod(shape)
        self.length = max(1, length if length is not None else 1)
        self.grad = grad
        self._cloned_init_net = None
        # Optionally store equivalent copies of the blob
        # in different precisions (i.e. half and float copies)
        # stored as a dict of TensorProto.DataType -> BlobReference
        self.blob_copy = blob_copy
        # each param_info can have its own optimizer. It can be set within
        # OptimizerContext (caffe2/python/optimizer.py)
        self._optimizer = None

    def grad_type(self):
        # self.grad could be None for model parallelism with parameter server
        if self.grad is None:
            return
        return (
            ParameterType.SPARSE if isinstance(self.grad, core.GradientSlice)
            else ParameterType.DENSE)

    @property
    def parameter(self):
        return self.blob

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        assert self._optimizer is None, "optimizer has already been set"
        self._optimizer = value

    def __str__(self):
        return self.name

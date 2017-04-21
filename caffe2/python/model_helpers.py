## @package model_helper_api
# Module caffe2.python.model_helper_api
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import copy

# flake8: noqa
from caffe2.python.helpers.dropout import *
from caffe2.python.helpers.arg_scope import *
from caffe2.python.helpers.fc import *
from caffe2.python.helpers.pooling import *
from caffe2.python.helpers.normalization import *
from caffe2.python.helpers.nonlinearity import *
from caffe2.python.helpers.array_helpers import *
from caffe2.python.helpers.algebra import *
from caffe2.python.helpers.train import *
from caffe2.python.helpers.conv import *


class HelperWrapper(object):
    _registry = {
        'arg_scope': arg_scope,
        'FC': FC,
        'PackedFC': PackedFC,
        'FC_Decomp': FC_Decomp,
        'FC_Sparse': FC_Sparse,
        'FC_Prune': FC_Prune,
        'Dropout': Dropout,
        'MaxPool': MaxPool,
        'AveragePool': AveragePool,
        'LRN': LRN,
        'Softmax': Softmax,
        'InstanceNorm': InstanceNorm,
        'SpatialBN': SpatialBN,
        'Relu': Relu,
        'PRelu': PRelu,
        'Concat': Concat,
        'DepthConcat': DepthConcat,
        'Sum': Sum,
        'Transpose': Transpose,
        'Iter': Iter,
        'Accuracy': Accuracy,
        'Conv': Conv,
        'ConvNd': ConvNd,
        'ConvTranspose': ConvTranspose,
        'GroupConv': GroupConv,
        'GroupConv_Deprecated': GroupConv_Deprecated,
    }

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getattr__(self, helper_name):
        if helper_name not in self._registry:
            raise AttributeError(
                "Helper function {} not "
                "registered.".format(helper_name)
            )

        def scope_wrapper(*args, **kwargs):
            cur_scope = get_current_scope()
            new_kwargs = copy.deepcopy(cur_scope.get(helper_name, {}))
            new_kwargs.update(kwargs)
            return self._registry[helper_name](*args, **new_kwargs)

        scope_wrapper.__name__ = helper_name
        return scope_wrapper

    def Register(self, helper):
        name = helper.__name__
        if name in self._registry:
            raise AttributeError(
                "Helper {} already exists. Please change your "
                "helper name.".format(name)
            )
        self._registry[name] = helper

    def has_helper(self, helper_or_helper_name):
        helper_name = (
            helper_or_helper_name
            if isinstance(helper_or_helper_name, basestring) else
            helper_or_helper_name.__name__
        )
        return helper_name in self._registry


sys.modules[__name__] = HelperWrapper(sys.modules[__name__])

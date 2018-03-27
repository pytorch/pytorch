## @package model_helper_api
# Module caffe2.python.model_helper_api
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import copy
import inspect
from past.builtins import basestring
from caffe2.python.model_helper import ModelHelper

# flake8: noqa
from caffe2.python.helpers.algebra import *
from caffe2.python.helpers.arg_scope import *
from caffe2.python.helpers.array_helpers import *
from caffe2.python.helpers.control_ops import *
from caffe2.python.helpers.conv import *
from caffe2.python.helpers.db_input import *
from caffe2.python.helpers.dropout import *
from caffe2.python.helpers.elementwise_linear import *
from caffe2.python.helpers.fc import *
from caffe2.python.helpers.nonlinearity import *
from caffe2.python.helpers.normalization import *
from caffe2.python.helpers.pooling import *
from caffe2.python.helpers.tools import *
from caffe2.python.helpers.train import *


class HelperWrapper(object):
    _registry = {
        'arg_scope': arg_scope,
        'fc': fc,
        'packed_fc': packed_fc,
        'fc_decomp': fc_decomp,
        'fc_sparse': fc_sparse,
        'fc_prune': fc_prune,
        'dropout': dropout,
        'max_pool': max_pool,
        'average_pool': average_pool,
        'max_pool_with_index' : max_pool_with_index,
        'lrn': lrn,
        'softmax': softmax,
        'instance_norm': instance_norm,
        'spatial_bn': spatial_bn,
        'relu': relu,
        'prelu': prelu,
        'tanh': tanh,
        'concat': concat,
        'depth_concat': depth_concat,
        'sum': sum,
        'transpose': transpose,
        'iter': iter,
        'accuracy': accuracy,
        'conv': conv,
        'conv_nd': conv_nd,
        'conv_transpose': conv_transpose,
        'group_conv': group_conv,
        'group_conv_deprecated': group_conv_deprecated,
        'image_input': image_input,
        'video_input': video_input,
        'add_weight_decay': add_weight_decay,
        'elementwise_linear': elementwise_linear,
        'layer_norm': layer_norm,
        'batch_mat_mul' : batch_mat_mul,
        'cond' : cond,
        'loop' : loop,
        'db_input' : db_input,
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
            new_kwargs = {}
            if helper_name != 'arg_scope':
                if len(args) > 0 and isinstance(args[0], ModelHelper):
                    model = args[0]
                elif 'model' in kwargs:
                    model = kwargs['model']
                else:
                    raise RuntimeError(
                "The first input of helper function should be model. " \
                "Or you can provide it in kwargs as model=<your_model>.")
                new_kwargs = copy.deepcopy(model.arg_scope)
            func = self._registry[helper_name]
            var_names, _, varkw, _= inspect.getargspec(func)
            if varkw is None:
                # this helper function does not take in random **kwargs
                new_kwargs = {
                    var_name: new_kwargs[var_name]
                    for var_name in var_names if var_name in new_kwargs
                }

            cur_scope = get_current_scope()
            new_kwargs.update(cur_scope.get(helper_name, {}))
            new_kwargs.update(kwargs)
            return func(*args, **new_kwargs)

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

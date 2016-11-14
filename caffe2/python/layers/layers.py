from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.tags import TagContext

from collections import namedtuple
import numpy as np

# Some types to simplify descriptions of things traveling between ops
IdList = schema.List(np.int64)
IdScoreList = schema.Map(np.int64, np.float32)


class InstantiationContext(object):
    """
    List of contexts where layer could be instantitated
    """
    TRAINING = 'training'
    PREDICTION = 'prediction'


_LAYER_REGISTRY = {}


def register_layer(name, layer):
    assert name not in _LAYER_REGISTRY, "{0} already exists".format(name)
    _LAYER_REGISTRY[name] = layer


def layer_exists(name):
    return name in _LAYER_REGISTRY


def create_layer(name, *args, **kwargs):
    return _LAYER_REGISTRY[name](*args, **kwargs)

# TODO(amalevich): Modify this to some better struct, something closer to
# ParameterInfo.
LayerParameter = namedtuple(
    'LayerParameter', ['parameter', 'optimizer', 'initializer'])


class ModelLayer(object):

    def __init__(self, model, prefix, input_record, tags=set(), **kwargs):
        self.name = model.next_block_name(prefix)
        self.model = model
        self.kwargs = kwargs
        self.input_record = input_record
        self.output_schema = None
        self.tags = set(tags)
        self.tags.update(TagContext.current().tags)
        self.params = []

    def get_output_schema(self):
        assert self.output_schema is not None, "Schema is not initialized"
        return self.output_schema

    def get_parameters(self):
        return self.params

    def add_operators(self, net, init_net=None,
                      context=InstantiationContext.TRAINING):
        if context != InstantiationContext.PREDICTION:
            assert init_net,\
                "Only prediction context can be used without init_net"
        if init_net:
            for param in self.params:
                # TODO(amalevich): Either return back to lambdas, that add all
                # params (looks a bit safer and breaking less abstractions) or
                # extend Net interface to this type of operations better
                init_net._net.op.extend([param.initializer])
        if context == InstantiationContext.TRAINING:
            self.add_train_ops(net)
        else:
            self.add_ops(net)

    def add_ops(self, net):
        raise NotImplementedError

    def add_train_ops(self, net):
        # Default train layer implementation is completely matching predict
        # layer implementation.
        self.add_ops(net)

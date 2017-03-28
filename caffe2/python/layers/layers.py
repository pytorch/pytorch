from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema, scope
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


def get_layer_class(name):
    return _LAYER_REGISTRY[name]


def create_layer(layer_name, *args, **kwargs):
    return _LAYER_REGISTRY[layer_name](*args, **kwargs)


# TODO(amalevich): Modify this to some better struct, something closer to
# ParameterInfo.
LayerParameter = namedtuple(
    'LayerParameter', ['parameter', 'optimizer', 'initializer'])


def _is_request_only_scalar(scalar):
    if len(scalar.field_metadata()) == 0:
        return False
    for metadata in scalar.field_metadata():
        if not (metadata and metadata.feature_specs and getattr(
                metadata.feature_specs, 'feature_is_request_only', False)):
            return False
    return True


class ModelLayer(object):

    def __init__(self, model, prefix, input_record, tags=None, **kwargs):
        self.name = model.next_layer_name(prefix)
        self.model = model
        self.kwargs = kwargs
        self.input_record = input_record
        self.request_only = True
        if len(input_record.all_scalars()) == 0:
            self.request_only = False
        for scalar in input_record.all_scalars():
            if not _is_request_only_scalar(scalar):
                self.request_only = False
                break
        self.output_schema = None
        self.tags = set(tags or set())
        self.tags.update(TagContext.current().tags)
        self.params = []

    def get_type(self):
        return self.__class__.__name__

    def get_output_schema(self):
        assert self.output_schema is not None, "Schema is not initialized"
        return self.output_schema

    def get_parameters(self):
        return self.params

    def get_fp16_compatible_parameters(self):
        """Return a subset of parameters which can be converted to fp16"""
        return []

    def get_memory_usage(self):
        return 0

    def add_operators(self, net, init_net=None,
                      context=InstantiationContext.TRAINING):
        # Namescope below should warranty that all intermediate blobs will be
        # assiciated with the layer that produces them
        with scope.NameScope(self.name):
            if context != InstantiationContext.PREDICTION:
                assert init_net,\
                    "Only prediction context can be used without init_net"
            if init_net:
                for param in self.params:
                    # TODO(amalevich): Either return back to lambdas, that add
                    # all params (looks a bit safer and breaking less
                    # abstractions) or extend Net interface to this type of
                    # operations better
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

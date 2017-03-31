## @package layers
# Module caffe2.python.layers.layers
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
    EVAL = 'eval'
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


LayerPsParam = namedtuple('LayerPsParam', ['sparse_key', 'average_length'])


# TODO(amalevich): Modify this to some better struct, something closer to
# ParameterInfo.
LayerParameter = namedtuple(
    'LayerParameter',
    ['parameter', 'optimizer', 'initializer', 'ps_param'])
LayerParameter.__new__.__defaults__ = (None, None, None, None)


def _is_request_only_scalar(scalar):
    if len(scalar.field_metadata()) == 0:
        return False
    for metadata in scalar.field_metadata():
        if not (metadata and metadata.feature_specs and getattr(
                metadata.feature_specs, 'feature_is_request_only', False)):
            return False
    return True


class ModelLayer(object):

    def __init__(self, model, prefix, input_record,
                 predict_input_record_fields=None, tags=None, **kwargs):
        """
        Base class for model layers. Layer is an abstraction that allows to
        provide model description in terms of meta-operators, where each of the
        meta-operators can have different implementations for training,
        evaluation and prediction, that are instantiated later. As an example
        SampledSoftmax can do something related to sampling depending on
        supervision during the training and just apply softmax if it's used for
        prediction/evaluation.

        All inputs/outputs from layers are represented as a record (instance of
        schema bounded to blobs) and are accessible through input_record and
        output_schema. If Layer needs to have only a subset of inputs/provides
        subset of outputs during the inference - it should provide
        predict_input_record and predict_output_schema correspondingly (those
        records are expected to be a subset of input_record/output_schema).

        Each layer is also have list of Tags associated with it, that depends on
        current context and arguments. It's possible to use those tags during
        the instantiation time.

        """
        self.name = model.next_layer_name(prefix)
        self.model = model
        self.kwargs = kwargs
        self._input_record = input_record
        if predict_input_record_fields:
            if not isinstance(predict_input_record_fields, list):
                predict_input_record_fields = [predict_input_record_fields]
            self._predict_input_record = self._input_record[
                predict_input_record_fields]
        else:
            self._predict_input_record = None

        self.request_only = True
        if len(input_record.all_scalars()) == 0:
            self.request_only = False
        for scalar in input_record.all_scalars():
            if not _is_request_only_scalar(scalar):
                self.request_only = False
                break

        self._output_schema = None
        self._predict_output_schema = None
        self.eval_output_schema = None
        self.tags = set(tags or [])
        self.tags.update(TagContext.current().tags)
        self.params = []

    def get_type(self):
        return self.__class__.__name__

    def _check_output_schema(self):
        assert self._output_schema is not None, "Schema is not initialized"
        assert (self._predict_output_schema is None or
                schema.is_schema_subset(self._predict_output_schema,
                                        self._output_schema)), (
            "predict_output_schema is not a subset of the output_schema")

    @property
    def predict_input_record(self):
        return self._predict_input_record or self._input_record

    @property
    def input_record(self):
        return self._input_record

    @property
    def predict_output_schema(self):
        self._check_output_schema()
        return self._predict_output_schema or self._output_schema

    @predict_output_schema.setter
    def predict_output_schema(self, output_schema):
        assert self._predict_output_schema is None
        self._predict_output_schema = output_schema

    @property
    def output_schema(self):
        self._check_output_schema()
        return self._output_schema

    @output_schema.setter
    def output_schema(self, output_schema):
        assert self._output_schema is None
        self._output_schema = output_schema

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
            if context not in {InstantiationContext.PREDICTION,
                               InstantiationContext.EVAL}:
                assert init_net, (
                    "Only prediction and eval context don't need init_net")
            if init_net:
                for param in self.params:
                    # TODO(amalevich): Either return back to lambdas, that add
                    # all params (looks a bit safer and breaking less
                    # abstractions) or extend Net interface to this type of
                    # operations better
                    init_net._net.op.extend([param.initializer])
            if context == InstantiationContext.TRAINING:
                self.add_train_ops(net)
            elif context == InstantiationContext.EVAL:
                self.add_eval_ops(net)
            else:
                self.add_ops(net)

    def add_ops(self, net):
        raise NotImplementedError

    def add_eval_ops(self, net):
        # Default train layer implementation is completely matching predict
        # layer implementation.
        self.add_ops(net)

    def add_train_ops(self, net):
        # Default eval layer implementation is completely matching eval
        # layer implementation.
        self.add_eval_ops(net)

## @package layer_model_helper
# Module caffe2.python.layer_model_helper
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, model_helper, schema
from caffe2.python.layers import layers

import logging
import numpy as np
logger = logging.getLogger(__name__)


class LayerModelHelper(model_helper.ModelHelper):
    """
    Model helper for building models on top of layers abstractions.

    Each layer is the abstraction that is higher level than Operator. Layer
    is responsible for ownership of it's own parameters and can easily be
    instantiated in multiple nets possible with different sets of ops.
    As an example: one can easily instantiate predict and train nets from
    the same set of layers, where predict net will have subset of the
    operators from train net.
    """

    def __init__(self, name, input_feature_schema, trainer_extra_schema,
                 keep_blobs=False):
        super(LayerModelHelper, self).__init__(name=name)
        self._layer_names = set()
        self._layers = []

        # optimizer bookkeeping
        self.param_to_optim = {}

        self._default_optimizer = None
        self._loss = None
        self._output_schema = None

        # Connect Schema to self.net. That particular instance of schmea will be
        # use for generation of the Layers accross the network and would be used
        # for connection with Readers.
        self._input_feature_schema = schema.NewRecord(
            self.net,
            input_feature_schema
        ) if not keep_blobs else input_feature_schema.clone()
        self._trainer_extra_schema = schema.NewRecord(
            self.net,
            trainer_extra_schema
        ) if not keep_blobs else trainer_extra_schema.clone()
        self._metrics_schema = schema.Struct()

        self._init_global_constants()
        self.param_init_net = self.create_init_net('param_init_net')

    def add_metric_field(self, name, value):
        assert name not in self._metrics_schema.fields, (
            "Try to add metric field twice: {}".format(name))
        self._metrics_schema = self._metrics_schema + schema.Struct(
            (name, value)
        )

    def add_global_constant(self, name, array=None, dtype=None,
                            initializer=None):
        # This is global namescope for constants. They will be created in all
        # init_nets and there should be very few of them.
        assert name not in self.global_constants
        self.global_constants[name] = self.net.NextBlob(name)

        if array is not None:
            assert initializer is None,\
                "Only one from array and initializer should be specified"
            if dtype is None:
                array = np.array(array)
            else:
                array = np.array(array, dtype=dtype)

            # TODO: make GivenTensor generic
            op_name = None
            if array.dtype == np.int32:
                op_name = 'GivenTensorIntFill'
            elif array.dtype == np.int64:
                op_name = 'GivenTensorInt64Fill'
            elif array.dtype == np.str:
                op_name = 'GivenTensorStringFill'
            elif array.dtype == np.bool:
                op_name = 'GivenTensorBoolFill'
            else:
                op_name = 'GivenTensorFill'

            def initializer(blob_name):
                return core.CreateOperator(op_name,
                                           [],
                                           blob_name,
                                           shape=array.shape,
                                           values=array.flatten().tolist()
                                           )
        else:
            assert initializer is not None

        self.global_constant_initializers.append(
            initializer(self.global_constants[name]))
        return self.global_constants[name]

    def _init_global_constants(self):
        self.global_constants = {}
        self.global_constant_initializers = []
        self.add_global_constant('ONE', 1.0)
        self.add_global_constant('ZERO', 0.0)
        self.add_global_constant('ZERO_RANGE', [0, 0], dtype='int32')
        self.add_global_constant('OFFLINE_TRAINING', True, dtype='bool')

    def _add_global_constants(self, init_net):
        for initializer_op in self.global_constant_initializers:
            init_net._net.op.extend([initializer_op])

    def create_init_net(self, name):
        init_net = core.Net(name)
        self._add_global_constants(init_net)
        return init_net

    def next_layer_name(self, prefix):
        base_name = core.ScopedName(prefix)
        name = base_name
        index = 0
        while name in self._layer_names:
            name = base_name + '_auto_' + str(index)
            index += 1

        self._layer_names.add(name)
        return name

    def add_layer(self, layer):
        self._layers.append(layer)
        for param in layer.get_parameters():
            assert isinstance(param.parameter, core.BlobReference)

            self.param_to_optim[str(param.parameter)] = \
                param.optimizer or self.default_optimizer

        # The primary value of adding everything to self.net - generation of the
        # operators right away, i.e. if error happens it'll be detected
        # immediately. Other than this - create_x_net should be called.
        layer.add_operators(self.net, self.param_init_net)
        return layer.output_schema

    def get_parameter_blobs(self):
        param_blobs = []
        for layer in self._layers:
            for param in layer.get_parameters():
                param_blobs.append(param.parameter)

        return param_blobs

    @property
    def default_optimizer(self):
        return self._default_optimizer

    @default_optimizer.setter
    def default_optimizer(self, optimizer):
        self._default_optimizer = optimizer

    @property
    def input_feature_schema(self):
        return self._input_feature_schema

    @property
    def trainer_extra_schema(self):
        return self._trainer_extra_schema

    @property
    def metrics_schema(self):
        """
        Returns the schema that represents model output that should be used for
        metric reporting.

        During the training/evaluation this schema will be appended to the
        schema that represents model output.
        """
        return self._metrics_schema

    @property
    def output_schema(self):
        assert self._output_schema is not None
        return self._output_schema

    @output_schema.setter
    def output_schema(self, schema):
        assert self._output_schema is None
        self._output_schema = schema

    @property
    def loss(self):
        assert self._loss is not None
        return self._loss

    @loss.setter
    def loss(self, loss):
        assert self._loss is None
        self._loss = loss

    def __getattr__(self, layer):
        if layer.startswith('__'):
            raise AttributeError(layer)

        # TODO(amalevich): Add add support for ifbpy inline documentation
        if layers.layer_exists(layer):
            def wrapper(*args, **kwargs):
                return self.add_layer(
                    layers.create_layer(layer, self, *args, **kwargs))
            return wrapper
        elif core.IsOperator(layer):
            def wrapper(*args, **kwargs):
                def apply_operator(net, in_record, out_record, **kwargs):
                    # TODO(amalevich): Switch to net.operator as soon as it gets
                    # landed
                    net.__getattr__(layer)(in_record.field_blobs(),
                                           out_record.field_blobs(),
                                           **kwargs)

                if 'name' not in kwargs:
                    kwargs['name'] = layer
                return self.add_layer(
                    layers.create_layer('Functional',
                                        self, *args, function=apply_operator,
                                        **kwargs))
            return wrapper
        else:
            raise ValueError(
                "Tring to create non-registered layer: {0}".format(layer))

    @property
    def layers(self):
        return self._layers

    def apply_optimizers(self, train_net, train_init_net, grad_map):
        for param, optimizer in self.param_to_optim.items():
            assert optimizer is not None, \
                "default optimizer must have been set in add_layer"
            # note that not all params has gradient and thus we sent None if
            # gradient does not exist
            optimizer(
                train_net, train_init_net, param, grad_map.get(str(param)))

    def _GetOne(self):
        return self.global_constants['ONE']

    # An optimizer which allows us to do NO optimization
    def NoOptim(self, *args, **kwargs):
        pass

## @package fc
# Module caffe2.python.layers.fc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
    LayerParameter
)
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin
import math
import numpy as np


class FC(SamplingTrainableMixin, ModelLayer):

    def __init__(self, model, input_record, output_dims, weight_init=None,
                 bias_init=None, weight_optim=None, bias_optim=None, name='fc',
                 **kwargs):
        super(FC, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        assert len(input_record.field_types()[0].shape) > 0, (
            "FC expects limited dimensions of the input tensor")

        input_dims = input_record.field_types()[0].shape[0]
        assert input_dims > 0, (
            "FC expects input dimensions > 0, got {}".format(input_dims))

        self.output_schema = schema.Scalar(
            (np.float32, (output_dims, )),
            model.net.NextScopedBlob(name + '_output')
        )

        scale = math.sqrt(1.0 / input_dims)
        weight_init = weight_init if weight_init else (
            'UniformFill', {'min': -scale, 'max': scale})
        bias_init = bias_init if bias_init else (
            'UniformFill', {'min': -scale, 'max': scale})

        self.w = model.net.NextScopedBlob(name + "_w")
        self.b = model.net.NextScopedBlob(name + "_b")

        self.params.append(
            LayerParameter(
                parameter=self.w,
                initializer=core.CreateOperator(weight_init[0],
                                                [],
                                                self.w,
                                                shape=[output_dims, input_dims],
                                                **weight_init[1]
                                                ),
                optimizer=weight_optim))
        self.params.append(
            LayerParameter(
                parameter=self.b,
                initializer=core.CreateOperator(bias_init[0],
                                                [],
                                                self.b,
                                                shape=[output_dims, ],
                                                **bias_init[1]
                                                ),
                optimizer=bias_optim))

    def _add_ops(self, net, params):
        net.FC(self.input_record.field_blobs() + params,
               self.output_schema.field_blobs(), **self.kwargs)

    @property
    def param_blobs(self):
        return [self.w, self.b]

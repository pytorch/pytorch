## @package add_bias
# Module caffe2.python.layers.add_bias
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
    LayerParameter
)
import math
import numpy as np


class AddBias(ModelLayer):

    def __init__(self, model, input_record, bias_init=None,
                 bias_optim=None, name='add_bias'):
        super(AddBias, self).__init__(model, name, input_record)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        assert len(input_record.field_type().shape) > 0, (
            "AddBias expects limited dimensions of the input tensor")

        input_dims = input_record.field_type().shape[0]
        assert input_dims > 0, (
            "AddBias expects input dimensions > 0, got {}".format(input_dims))

        self.output_schema = schema.Scalar(
            (input_record.field_type().base, (input_dims, )),
            model.net.NextScopedBlob(name + '_output')
        )

        scale = math.sqrt(1.0 / input_dims)
        bias_init = bias_init if bias_init else (
            'UniformFill', {'min': -scale, 'max': scale})

        self.b = model.net.NextScopedBlob(name + "_b")

        self.params.append(
            LayerParameter(
                parameter=self.b,
                initializer=core.CreateOperator(bias_init[0],
                                                [],
                                                self.b,
                                                shape=[input_dims, ],
                                                **bias_init[1]
                                                ),
                optimizer=bias_optim))

    def add_ops(self, net):
        net.Add(self.input_record.field_blobs() + [self.b],
                self.output_schema.field_blobs(), broadcast=1)

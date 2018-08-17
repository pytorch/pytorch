from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer

import numpy as np


class LayerNormalization(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name='layer_normalization',
        scale_optim=None,
        bias_optim=None,
        epsilon=1e-4,
        axis=1,
        **kwargs
    ):
        super(LayerNormalization, self).__init__(
            model, name, input_record, **kwargs)

        assert isinstance(input_record, schema.Scalar), (
            "Incorrect input type: {}".format(input_record))

        self.input_shape = input_record.field_type().shape
        self.epsilon = epsilon
        self.axis = axis

        assert len(self.input_shape) >= 1, (
            "This layer supports only >= 2D tesnors")
        input_dims = self.input_shape[0]

        self.output_schema = schema.Scalar(
            (np.float32, self.input_shape),
            self.get_next_blob_reference('output')
        )

        self.scale = self.create_param(param_name='scale',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': 1.0}),
                                       optimizer=scale_optim)
        self.bias = self.create_param(param_name='bias',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': 0.0}),
                                       optimizer=bias_optim)

    def add_ops(self, net):
        input_blob = self.input_record.field_blobs()
        ln_output = self.output_schema.field_blobs()

        output_blobs = [net.NextScopedBlob('ln_output'), net.NextScopedBlob('ln_mean'),
                        net.NextScopedBlob('ln_stdev')]

        normalized, mean, stdev = net.LayerNorm(input_blob,
            output_blobs,
            axis=self.axis,
            epsilon=self.epsilon)

        scaled = net.Mul(
            [normalized, self.scale],
            [net.NextScopedBlob('ln_scaled')],
            broadcast=1,
            axis=self.axis,
        )

        net.Add(
            [scaled, self.bias],
            ln_output,
            broadcast=1,
            axis=self.axis,
        )

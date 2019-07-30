## @package fc_with_batchmatmul
# Module caffe2.python.layers.fc_with_batchmatmul
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin

import math
import numpy as np


class FcWithBatchMatMul(SamplingTrainableMixin, ModelLayer):
    '''Fc layer enabliing input to be a high-rank tensor. If the input is a matrix,
    this layer would be reduced into a regular fc layer.
    This layer takes input with shape (batch_size, x1, x2, ..., xn, input_dim),
    creates weight matrix w with shape (x1, x2, ..., xn, output_dim, input_dim)
    and bias with shape (x1, x2, ..., xn, output_dim)
    Output has shape (batch_size, x1, x2, ..., xn, output_dim).
    '''
    def __init__(
        self,
        model,
        input_record,
        output_dims,
        weight_init=None,
        bias_init=None,
        weight_optim=None,
        bias_optim=None,
        weight_reg=None,
        bias_reg=None,
        name='fc_with_batchmatmul',
        **kwargs
    ):
        super(FcWithBatchMatMul, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        assert len(input_record.field_types()[0].shape) > 0, (
            "FcWithBatchMatMul expects limited dimensions of the input tensor"
        )

        self.input_shape = list(input_record.field_type().shape)
        if len(self.input_shape) > 1:
            self.input_shape = [self.input_shape[-2]] + self.input_shape[:-2] + self.input_shape[-1:]

        input_dims = self.input_shape[-1]
        assert input_dims > 0, (
            "FcWithBatchMatMul expects input dimensions > 0, got {}".format(input_dims)
        )

        batch_dims = self.input_shape[:-1]

        self.output_schema = schema.Scalar(
            (np.float32, batch_dims + [output_dims, ]),
            self.get_next_blob_reference('output')
        )

        scale = math.sqrt(1.0 / input_dims)
        weight_init = weight_init if weight_init else (
            'UniformFill', {'min': -scale,
                            'max': scale}
        )
        bias_init = bias_init if bias_init else (
            'UniformFill', {'min': -scale, 'max': scale})

        self.w = self.create_param(
            param_name='w',
            shape=batch_dims + [output_dims, input_dims],
            initializer=weight_init,
            optimizer=weight_optim,
            regularizer=weight_reg
        )
        if len(batch_dims) > 1:
            batch_dims = batch_dims[1:] + [batch_dims[0]]
        self.b = self.create_param(
            param_name='b',
            shape=batch_dims + [output_dims, ],
            initializer=bias_init,
            optimizer=bias_optim,
            regularizer=bias_reg
        )

    def _add_ops(self, net, params):
        transpose_axis = list(range(len(self.input_shape) + 1))
        transpose_axis[0], transpose_axis[-2] = transpose_axis[-2], transpose_axis[0]
        output_blob_transpose1 = net.NextScopedBlob(
            'output_transpose')
        input_record = net.Transpose(
            self.input_record.field_blobs(), [output_blob_transpose1], axes=transpose_axis
        )
        output_blob_w = net.NextScopedBlob(
            'output_w')
        wx = net.BatchMatMul(
            [input_record] + params[:1],
            [output_blob_w],
            trans_b=1,
            **self.kwargs
        )
        output_blob_transpose2 = net.NextScopedBlob(
            'output_transpose2')
        wx = net.Transpose(
            [wx], [output_blob_transpose2], axes=transpose_axis
        )
        net.Add(
            [wx] + params[1:],
            self.output_schema.field_blobs(), **self.kwargs
        )


    @property
    def param_blobs(self):
        return [self.w, self.b]

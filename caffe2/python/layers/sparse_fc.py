# @package sparse_fc
# Module caffe2.python.layers.sparse_fc
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin

import math
import numpy as np


class SparseFC(SamplingTrainableMixin, ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        output_dims,
        sparse_idx,
        weight_init=None,
        bias_init=None,
        weight_optim=None,
        bias_optim=None,
        weight_reg=None,
        bias_reg=None,
        axis=1,
        name='sparse_fc',
        **kwargs
    ):
        super(SparseFC, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), (
            "Incorrect input type {}".format(input_record))

        self.input_shape = input_record.dtype.shape
        sparse_idx_shape = sparse_idx.dtype.shape
        self.sparse_idx = sparse_idx
        self.axis = axis
        assert len(self.input_shape) >= 1, (
            "SparseFC expects at least 2 dimensions of the input tensor (including batch_size, if any)")
        assert axis >= 1, "axis {} should >= 1.".format(axis)

        assert len(sparse_idx_shape[axis - 1:]) == 1, (
            "sparse_idx[%d - 1:] must be of length 1, but got {}".format(
                sparse_idx_shape[axis - 1:]
            )
        )
        assert (
            self.input_shape[:axis - 1] == sparse_idx_shape[:axis - 1]
        ), "Outer dimensions of INPUT and SPARSE_IDX must be the same, but got {} and {}".format(
            self.input_shape[:axis - 1], sparse_idx_shape[:axis - 1],
        )
        input_inner_dims = self.input_shape[axis - 1:]
        assert len(input_inner_dims) == 1 or len(input_inner_dims) == 2, (
            "input_record's inner dimensions must be 1 or 2 beyond axis=%d, but got {}".format(
                axis, self.input_shape[axis - 1:]
            )
        )

        self.last_dim = self.input_shape[-1] if len(input_inner_dims) == 2 else 1

        input_dims = self.input_shape[axis - 1]
        assert input_dims > 0, (
            "FC expects input dimensions > 0, got {}".format(input_dims))


        num_nonzeros = sparse_idx_shape[0]

        scale = math.sqrt(1.0 / input_dims)
        weight_init = weight_init if weight_init else (
            'UniformFill', {'min': -scale, 'max': scale})
        bias_init = bias_init if bias_init else (
            'UniformFill', {'min': -scale, 'max': scale})

        self.w = self.create_param(param_name='w',
                                   shape=[output_dims, input_dims],
                                   initializer=weight_init,
                                   optimizer=weight_optim,
                                   regularizer=weight_reg)

        self.b = self.create_param(param_name='b',
                                   shape=[output_dims, ],
                                   initializer=bias_init,
                                   optimizer=bias_optim,
                                   regularizer=bias_reg)

        output_shape = list(self.input_shape)[0:axis - 1] + [output_dims]
        if self.last_dim > 1:
            output_shape = tuple(output_shape + [self.last_dim])

        self.output_schema = schema.Scalar(
            (np.float32, output_shape),
            self.get_next_blob_reference('output'),
        )

    def _add_ops(self, net, params):

        input = self.input_record.field_blobs()[0]
        if self.last_dim == 1:
            input = net.ExpandDims(
                input, "input_expanded", dims=[len(self.input_shape)]
            )

        # (batch_size, num_nonzeros, self.last_dim)
        gathered_input = net.BatchGather(
            [input] + self.sparse_idx.field_blobs(), "gathered_input", match_outer=True
        )

        # self.w is of shape (output_dims, input_dims)
        # after gather with idx of shape (batch_size, num_nonzeros) along axis=1
        # it becomes (output_dims, batch_size, num_nonzeros)
        gathered_w = net.Gather(
            [self.w] + self.sparse_idx.field_blobs(), "gathered_w", axis=1,
        )

        # (batch_size, output_dims, num_nonzeros)
        gathered_w_trans = net.Transpose(
            gathered_w, "gathered_w_trans", axes=[1, 0, 2],
        )

        # (batch_size, output_dims, self.last_dim)
        output_no_bias = net.BatchMatMul(
            [gathered_w_trans, gathered_input], "output_no_bias"
        )

        if self.last_dim == 1:  # squeeze back
            output_no_bias = net.Squeeze(
                output_no_bias,
                "output_no_bias_last_dim_squeezed",
                dims=[len(self.input_shape)]
            )

        net.Add(
            [output_no_bias, self.b],
            self.output_schema.field_blobs(), axis=self.axis, broadcast=1
        )

    @property
    def param_blobs(self):
        return [self.w, self.b]

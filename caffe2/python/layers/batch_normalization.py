from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
    LayerParameter
)

import numpy as np


class BatchNormalization(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name='batch_normalization',
        scale_optim=None,
        bias_optim=None,
        momentum=0.9,
        order='NCHW',
        **kwargs
    ):
        super(BatchNormalization, self).__init__(
            model, name, input_record, **kwargs)

        assert isinstance(input_record, schema.Scalar), "Incorrect input type"

        self.input_shape = input_record.field_type().shape

        if len(self.input_shape) == 3:
            if order == "NCHW":
                input_dims = self.input_shape[0]
            elif order == "NHWC":
                input_dims = self.input_shape[2]
            else:
                raise ValueError("Please specify a correct order")
        else:
            assert len(self.input_shape) == 1, (
                "This layer supports only 4D or 2D tesnors")
            input_dims = self.input_shape[0]

        self.output_schema = schema.Scalar(
            (np.float32, self.input_shape),
            model.net.NextScopedBlob(name + '_output')
        )

        self.momentum = momentum
        self.order = order

        self.scale = model.net.NextScopedBlob(name + "_scale")
        self.bias = model.net.NextScopedBlob(name + "_bias")
        self.rm = model.net.NextScopedBlob(name + "_running_mean")
        self.riv = model.net.NextScopedBlob(name + "_running_inv_var")

        self.params.append(
            LayerParameter(
                parameter=self.scale,
                initializer=core.CreateOperator('ConstantFill',
                                                [],
                                                self.scale,
                                                shape=[input_dims],
                                                value=1.0,
                                                ),
                optimizer=scale_optim))
        self.params.append(
            LayerParameter(
                parameter=self.bias,
                initializer=core.CreateOperator('ConstantFill',
                                                [],
                                                self.bias,
                                                shape=[input_dims],
                                                value=0.0,
                                                ),
                optimizer=bias_optim))
        self.params.append(
            LayerParameter(
                parameter=self.rm,
                initializer=core.CreateOperator('ConstantFill',
                                                [],
                                                self.rm,
                                                shape=[input_dims],
                                                value=0.0,
                                                ),
                optimizer=model.NoOptim))
        self.params.append(
            LayerParameter(
                parameter=self.riv,
                initializer=core.CreateOperator('ConstantFill',
                                                [],
                                                self.riv,
                                                shape=[input_dims],
                                                vlaue=1.0,
                                                ),
                optimizer=model.NoOptim))

    def _add_ops(self, net, is_test):
        input_blob = self.input_record.field_blobs()[0]
        if len(self.input_shape) == 1:
            input_blob = net.ExpandDims(input_blob,
                                        input_blob,
                                        dims=[2, 3])

        bn_output = self.output_schema.field_blobs()
        if is_test:
            output_blobs = bn_output
        else:
            output_blobs = bn_output + [self.rm, self.riv,
                                        net.NextScopedBlob('bn_saved_mean'),
                                        net.NextScopedBlob('bn_saved_iv')]

        net.SpatialBN([input_blob, self.scale,
                       self.bias, self.rm, self.riv],
                      output_blobs,
                      momentum=self.momentum,
                      is_test=is_test,
                      order=self.order)

        if len(self.input_shape) == 1:
            net.Squeeze(bn_output,
                        bn_output,
                        dims=[2, 3])

    def add_train_ops(self, net):
        self._add_ops(net, is_test=False)

    def add_eval_ops(self, net):
        self._add_ops(net, is_test=True)

    def add_ops(self, net):
        self.add_eval_ops(net)

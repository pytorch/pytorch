




from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer

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
        scale_init_value=1.0,
        **kwargs
    ):
        super().__init__(model, name, input_record, **kwargs)

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
                "This layer supports only 4D or 2D tensors")
            input_dims = self.input_shape[0]

        self.output_schema = schema.Scalar(
            (np.float32, self.input_shape),
            self.get_next_blob_reference('output')
        )

        self.momentum = momentum
        self.order = order

        self.scale = self.create_param(param_name='scale',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': scale_init_value}),
                                       optimizer=scale_optim)
        self.bias = self.create_param(param_name='bias',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': 0.0}),
                                       optimizer=bias_optim)
        self.rm = self.create_param(param_name='running_mean',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': 0.0}),
                                       optimizer=model.NoOptim)
        self.riv = self.create_param(param_name='running_inv_var',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': 1.0}),
                                       optimizer=model.NoOptim)

    def _add_ops(self, net, is_test, out_blob=None):
        original_input_blob = self.input_record.field_blobs()
        input_blob = net.NextScopedBlob('expand_input')
        if len(self.input_shape) == 1:
            input_blob = net.ExpandDims(original_input_blob,
                                        dims=[2, 3])
        else:
            input_blob = original_input_blob[0]

        if out_blob is None:
            bn_output = self.output_schema.field_blobs()
        else:
            bn_output = out_blob
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

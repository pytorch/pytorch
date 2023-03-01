




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
        use_layer_norm_op=True,
        scale_init_value=1.0,
        **kwargs
    ):
        super().__init__(model, name, input_record, **kwargs)

        assert isinstance(input_record, schema.Scalar), (
            "Incorrect input type: {}".format(input_record))

        self.input_shape = input_record.field_type().shape
        self.axis = axis

        assert len(self.input_shape) >= 1, (
            "This layer supports only >= 2D tensors")
        input_dims = self.input_shape[0]

        self.output_schema = schema.Scalar(
            (np.float32, self.input_shape),
            self.get_next_blob_reference('output')
        )

        self.scale = self.create_param(param_name='scale',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': scale_init_value}),
                                       optimizer=scale_optim)
        self.bias = self.create_param(param_name='bias',
                                       shape=[input_dims],
                                       initializer=('ConstantFill', {'value': 0.0}),
                                       optimizer=bias_optim)
        self.use_layer_norm_op = use_layer_norm_op

        if self.use_layer_norm_op:
            self.epsilon = epsilon
        else:
            assert len(self.input_shape) == 1, (
                "When using alternative implementation, "
                "input data can only be 2D"
            )
            self.epsilon = model.maybe_add_global_constant(
                "%s_epsilon" % self.name, float(epsilon)
            )

    def add_ops_with_layer_norm_op(self, net):
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

    def add_ops_without_layer_norm_op(self, net):
        # two issues here:
        #  1. use multiple ops to replace the function of LayerNorm
        #  2. do not use legacy broadcast
        ln_output = net.NextScopedBlob("ln_output")
        ln_mean = net.NextScopedBlob("ln_mean")
        ln_stdev = net.NextScopedBlob("ln_stdev")
        ln_mean_arr = net.NextScopedBlob("ln_mean_arr")
        net.ReduceBackMean(self.input_record.field_blobs(), [ln_mean_arr])
        net.ExpandDims([ln_mean_arr], [ln_mean], dims=[1])
        ln_centered = net.NextScopedBlob("ln_centered")
        net.Sub(self.input_record.field_blobs() + [ln_mean], [ln_centered])
        ln_sqr = net.NextScopedBlob("ln_sqr")
        net.Sqr([ln_centered], [ln_sqr])
        ln_sqr_mean = net.NextScopedBlob("ln_sqr_mean")
        net.ReduceBackMean([ln_sqr], [ln_sqr_mean])
        ln_var = net.NextScopedBlob("ln_var")
        net.Add([ln_sqr_mean, self.epsilon], ln_var)
        ln_std_arr = net.NextScopedBlob("ln_std_arr")
        net.Pow([ln_var], [ln_std_arr], exponent=0.5)
        net.ExpandDims([ln_std_arr], [ln_stdev], dims=[1])
        net.Div([ln_centered, ln_stdev], [ln_output])
        ln_scaled = net.NextScopedBlob("ln_scaled")
        net.Mul([ln_output, self.scale], [ln_scaled])
        net.Add([ln_scaled, self.bias], self.output_schema.field_blobs())

    def add_ops(self, net):
        if self.use_layer_norm_op:
            self.add_ops_with_layer_norm_op(net)
        else:
            self.add_ops_without_layer_norm_op(net)

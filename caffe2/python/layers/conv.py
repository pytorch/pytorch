## @package conv
# Module caffe2.python.layers.conv





from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
import numpy as np


class Conv(ModelLayer):
    """
        Convolutional layer
        Input:
        - input_record: at least has the shape info of C (num_channels)
        - output_dim: number of convolutional filters
        - kernel_h, kernel_w: kernel size for h and w
        - stride_h, stride_w: stride for h and w
        - pad_b, pad_l, pad_r, pad_t: padding sizes, if stride == 1,
                                      'None' value will do auto padding
        - order: either 'NHWC' or 'NCHW'
    """

    def __init__(self, model, input_record, output_dim, kernel_h, kernel_w,
                 stride_h, stride_w, pad_b=None, pad_l=None, pad_r=None,
                 pad_t=None, order='NHWC', kernel_init=None, bias_init=None,
                 kernel_optim=None, bias_optim=None,
                 name='conv', **kwargs):

        super(Conv, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        # input num_channels (C) is needed
        input_dims = input_record.field_type().shape

        assert (kernel_h > 0 and isinstance(kernel_h, int)), (
            "kernel_h should be positive integer")
        assert (kernel_w > 0 and isinstance(kernel_w, int)), (
            "kernel_w should be positive integer")
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w

        assert (stride_h > 0 and isinstance(stride_h, int)), (
            "stride_h should be positive integer")
        assert (stride_w > 0 and isinstance(stride_w, int)), (
            "stride_w should be positive integer")
        self.stride_h = stride_h
        self.stride_w = stride_w

        # output_dim calculation (http://cs231n.github.io/convolutional-networks/)
        # output_dim_w = (input_dim_w - kernel_w + pad_r + pad_l) / stride_w + 1
        # so, do auto_padding requires
        # pad_r, pad_l = [(input_dim_w - 1) * stride_w - input_dim_w + kernel_w] / 2
        # similair for pad_t and pad_b to auto pad kernel_h
        # here we only do auto padding for stride = 1 case
        if stride_h == 1:
            pad_t = int((kernel_h - 1) / 2) if pad_t is None else pad_t
            pad_b = int((kernel_h - 1) / 2) if pad_b is None else pad_b
        else:
            pad_t = 0 if pad_t is None else pad_t
            pad_b = 0 if pad_b is None else pad_b

        if stride_w == 1:
            pad_r = int((kernel_w - 1) / 2) if pad_r is None else pad_r
            pad_l = int((kernel_w - 1) / 2) if pad_l is None else pad_l
        else:
            pad_r = 0 if pad_r is None else pad_r
            pad_l = 0 if pad_l is None else pad_l

        assert (pad_t >= 0 and isinstance(pad_t, int)), "pad_t should be int >= 0"
        assert (pad_b >= 0 and isinstance(pad_b, int)), "pad_b should be int >= 0"
        assert (pad_r >= 0 and isinstance(pad_r, int)), "pad_r should be int >= 0"
        assert (pad_l >= 0 and isinstance(pad_l, int)), "pad_l should be int >= 0"
        self.pad_t = pad_t
        self.pad_b = pad_b
        self.pad_r = pad_r
        self.pad_l = pad_l

        assert order in ['NHWC', 'NCHW'], "order should either 'NHWC' or 'NCHW'"
        self.order = order

        if order == 'NHWC':
            input_c = input_dims[-1]
            kernel_shape = [output_dim, kernel_h, kernel_w, input_c]
        elif order == 'NCHW':
            input_c = input_dims[0]
            kernel_shape = [output_dim, input_c, kernel_h, kernel_w]
        assert input_c > 0, (
            "Number of input channels in conv parameters should be positive")

        kernel_init = kernel_init if kernel_init else (
            'XavierFill', {}
        )
        bias_init = bias_init if bias_init else (
            'ConstantFill', {'value': 0.0}
        )

        self.kernel = self.create_param(
            param_name='conv_kernel',
            shape=kernel_shape,
            initializer=kernel_init,
            optimizer=kernel_optim,
        )

        self.bias = self.create_param(
            param_name='conv_bias',
            shape=[output_dim],
            initializer=bias_init,
            optimizer=bias_optim,
        )

        # the output_schema only has the num of output channels
        # output_h and output_w would be inferred internally
        self.output_schema = schema.Scalar(
            (np.float32, (output_dim,)),
            self.get_next_blob_reference('output')
        )

    def add_ops(self, net):
        net.Conv(
            self.input_record.field_blobs() + [self.kernel, self.bias],
            self.output_schema.field_blobs(),
            kernel_h=self.kernel_h,
            kernel_w=self.kernel_w,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
            pad_t=self.pad_t,
            pad_l=self.pad_l,
            pad_b=self.pad_b,
            pad_r=self.pad_r,
            order=self.order
        )

## @package cnn
# Module caffe2.python.cnn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, scope, model_helpers
from caffe2.python.model_helper import ModelHelperBase
from caffe2.proto import caffe2_pb2


class CNNModelHelper(ModelHelperBase):
    """A helper model so we can write CNN models more easily, without having to
    manually define parameter initializations and operators separately.
    """

    def __init__(self, order="NCHW", name=None,
                 use_cudnn=True, cudnn_exhaustive_search=False,
                 ws_nbytes_limit=None, init_params=True,
                 skip_sparse_optim=False,
                 param_model=None):

        super(CNNModelHelper, self).__init__(
            skip_sparse_optim=skip_sparse_optim,
            name="CNN" if name is None else name,
            init_params=init_params,
            param_model=param_model,
        )

        self.order = order
        self.use_cudnn = use_cudnn
        self.cudnn_exhaustive_search = cudnn_exhaustive_search
        self.ws_nbytes_limit = ws_nbytes_limit
        if self.order != "NHWC" and self.order != "NCHW":
            raise ValueError(
                "Cannot understand the CNN storage order %s." % self.order
            )

    def GetWeights(self, namescope=None):
        if namescope is None:
            namescope = scope.CurrentNameScope()

        if namescope == '':
            return self.weights[:]
        else:
            return [w for w in self.weights if w.GetNameScope() == namescope]

    def GetBiases(self, namescope=None):
        if namescope is None:
            namescope = scope.CurrentNameScope()

        if namescope == '':
            return self.biases[:]
        else:
            return [b for b in self.biases if b.GetNameScope() == namescope]

    def ImageInput(
            self, blob_in, blob_out, use_gpu_transform=False, **kwargs
    ):
        """Image Input."""
        if self.order == "NCHW":
            if (use_gpu_transform):
                kwargs['use_gpu_transform'] = 1 if use_gpu_transform else 0
                # GPU transform will handle NHWC -> NCHW
                data, label = self.net.ImageInput(
                    blob_in, [blob_out[0], blob_out[1]], **kwargs)
                # data = self.net.Transform(data, blob_out[0], **kwargs)
                pass
            else:
                data, label = self.net.ImageInput(
                    blob_in, [blob_out[0] + '_nhwc', blob_out[1]], **kwargs)
                data = self.net.NHWC2NCHW(data, blob_out[0])
        else:
            data, label = self.net.ImageInput(
                blob_in, blob_out, **kwargs)
        return data, label

    def _ConvBase(  # noqa
        self, is_nd, blob_in, blob_out, dim_in, dim_out, kernel,
        weight_init=None, bias_init=None, group=1, transform_inputs=None,
        **kwargs
    ):
        kernels = []
        if is_nd:
            if not isinstance(kernel, list):
                kernels = [kernel]
            else:
                kernels = kernel
        else:
            kernels = [kernel] * 2

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        use_bias =\
            False if ("no_bias" in kwargs and kwargs["no_bias"]) else True
        weight_init = weight_init if weight_init else ('XavierFill', {})
        bias_init = bias_init if bias_init else ('ConstantFill', {})
        blob_out = blob_out or self.net.NextName()
        weight_shape = [dim_out]
        if self.order == "NCHW":
            weight_shape.append(int(dim_in / group))
            weight_shape.extend(kernels)
        else:
            weight_shape.extend(kernels)
            weight_shape.append(int(dim_in / group))

        if self.init_params:
            weight = self.param_init_net.__getattr__(weight_init[0])(
                [],
                blob_out + '_w',
                shape=weight_shape,
                **weight_init[1]
            )
            if use_bias:
                bias = self.param_init_net.__getattr__(bias_init[0])(
                    [],
                    blob_out + '_b',
                    shape=[dim_out, ],
                    **bias_init[1]
                )
        else:
            weight = core.ScopedBlobReference(
                blob_out + '_w', self.param_init_net)
            if use_bias:
                bias = core.ScopedBlobReference(
                    blob_out + '_b', self.param_init_net)
        if use_bias:
            self.params.extend([weight, bias])
        else:
            self.params.extend([weight])

        self.weights.append(weight)

        if use_bias:
            self.biases.append(bias)

        if use_bias:
            inputs = [blob_in, weight, bias]
        else:
            inputs = [blob_in, weight]

        if transform_inputs is not None:
            transform_inputs(self, blob_out, inputs)

        # For the operator, we no longer need to provide the no_bias field
        # because it can automatically figure this out from the number of
        # inputs.
        if 'no_bias' in kwargs:
            del kwargs['no_bias']
        if group != 1:
            kwargs['group'] = group
        return self.net.Conv(
            inputs,
            blob_out,
            kernels=kernels,
            order=self.order,
            **kwargs)

    def ConvNd(self, blob_in, blob_out, dim_in, dim_out, kernel,
               weight_init=None, bias_init=None, group=1, transform_inputs=None,
               **kwargs):
        """N-dimensional convolution for inputs with NCHW storage order.
        """
        assert self.order == "NCHW", "ConvNd only supported for NCHW storage."
        return self._ConvBase(True, blob_in, blob_out, dim_in, dim_out, kernel,
                              weight_init, bias_init, group, transform_inputs,
                              **kwargs)

    def Conv(self, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None,
             bias_init=None, group=1, transform_inputs=None, **kwargs):
        """2-dimensional convolution.
        """
        return self._ConvBase(False, blob_in, blob_out, dim_in, dim_out, kernel,
                              weight_init, bias_init, group, transform_inputs,
                              **kwargs)

    def ConvTranspose(
        self, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None,
        bias_init=None, **kwargs
    ):
        """ConvTranspose.
        """
        weight_init = weight_init if weight_init else ('XavierFill', {})
        bias_init = bias_init if bias_init else ('ConstantFill', {})
        blob_out = blob_out or self.net.NextName()
        weight_shape = (
            [dim_in, dim_out, kernel, kernel]
            if self.order == "NCHW" else [dim_in, kernel, kernel, dim_out]
        )
        if self.init_params:
            weight = self.param_init_net.__getattr__(weight_init[0])(
                [],
                blob_out + '_w',
                shape=weight_shape,
                **weight_init[1]
            )
            bias = self.param_init_net.__getattr__(bias_init[0])(
                [],
                blob_out + '_b',
                shape=[dim_out, ],
                **bias_init[1]
            )
        else:
            weight = core.ScopedBlobReference(
                blob_out + '_w', self.param_init_net)
            bias = core.ScopedBlobReference(
                blob_out + '_b', self.param_init_net)
        self.params.extend([weight, bias])
        self.weights.append(weight)
        self.biases.append(bias)
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit
        return self.net.ConvTranspose(
            [blob_in, weight, bias],
            blob_out,
            kernel=kernel,
            order=self.order,
            **kwargs
        )

    def GroupConv(
        self,
        blob_in,
        blob_out,
        dim_in,
        dim_out,
        kernel,
        weight_init=None,
        bias_init=None,
        group=1,
        **kwargs
    ):
        """Group Convolution.

        This is essentially the same as Conv with a group argument passed in.
        We specialize this for backward interface compatibility.
        """
        return self.Conv(blob_in, blob_out, dim_in, dim_out, kernel,
                         weight_init=weight_init, bias_init=bias_init,
                         group=group, **kwargs)

    def GroupConv_Deprecated(
        self,
        blob_in,
        blob_out,
        dim_in,
        dim_out,
        kernel,
        weight_init=None,
        bias_init=None,
        group=1,
        **kwargs
    ):
        """GroupConvolution's deprecated interface.

        This is used to simulate a group convolution via split and concat. You
        should always use the new group convolution in your new code.
        """
        weight_init = weight_init if weight_init else ('XavierFill', {})
        bias_init = bias_init if bias_init else ('ConstantFill', {})
        use_bias = False if ("no_bias" in kwargs and kwargs["no_bias"]) else True
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit
        if dim_in % group:
            raise ValueError("dim_in should be divisible by group.")
        if dim_out % group:
            raise ValueError("dim_out should be divisible by group.")
        splitted_blobs = self.net.DepthSplit(
            blob_in,
            ['_' + blob_out + '_gconv_split_' + str(i) for i in range(group)],
            dimensions=[int(dim_in / group) for i in range(group)],
            order=self.order
        )
        weight_shape = (
            [dim_out / group, dim_in / group, kernel, kernel]
            if self.order == "NCHW" else
            [dim_out / group, kernel, kernel, dim_in / group]
        )
        # Make sure that the shapes are of int format. Especially for py3 where
        # int division gives float output.
        weight_shape = [int(v) for v in weight_shape]
        conv_blobs = []
        for i in range(group):
            if self.init_params:
                weight = self.param_init_net.__getattr__(weight_init[0])(
                    [],
                    blob_out + '_gconv_%d_w' % i,
                    shape=weight_shape,
                    **weight_init[1]
                )
                if use_bias:
                    bias = self.param_init_net.__getattr__(bias_init[0])(
                        [],
                        blob_out + '_gconv_%d_b' % i,
                        shape=[int(dim_out / group)],
                        **bias_init[1]
                    )
            else:
                weight = core.ScopedBlobReference(
                    blob_out + '_gconv_%d_w' % i, self.param_init_net)
                if use_bias:
                    bias = core.ScopedBlobReference(
                        blob_out + '_gconv_%d_b' % i, self.param_init_net)
            if use_bias:
                self.params.extend([weight, bias])
            else:
                self.params.extend([weight])
            self.weights.append(weight)
            if use_bias:
                self.biases.append(bias)
            if use_bias:
                inputs = [weight, bias]
            else:
                inputs = [weight]
            if 'no_bias' in kwargs:
                del kwargs['no_bias']
            conv_blobs.append(
                splitted_blobs[i].Conv(
                    inputs,
                    blob_out + '_gconv_%d' % i,
                    kernel=kernel,
                    order=self.order,
                    **kwargs
                )
            )
        concat, concat_dims = self.net.Concat(
            conv_blobs,
            [blob_out, "_" + blob_out + "_concat_dims"],
            order=self.order
        )
        return concat

    def FC(self, *args, **kwargs):
        return model_helpers.FC(self, *args, **kwargs)

    def PackedFC(self, *args, **kwargs):
        return model_helpers.PackedFC(self, *args, **kwargs)

    def FC_Prune(self, *args, **kwargs):
        return model_helpers.FC_Prune(self, *args, **kwargs)

    def FC_Decomp(self, *args, **kwargs):
        return model_helpers.FC_Decomp(self, *args, **kwargs)

    def FC_Sparse(self, *args, **kwargs):
        return model_helpers.FC_Sparse(self, *args, **kwargs)

    def Dropout(self, *args, **kwargs):
        return model_helpers.Dropout(self, *args, **kwargs)

    def LRN(self, *args, **kwargs):
        return model_helpers.LRN(self, *args, **kwargs)

    def Softmax(self, *args, **kwargs):
        return model_helpers.Softmax(self, *args, use_cudnn=self.use_cudnn,
                                     **kwargs)

    def SpatialBN(self, *args, **kwargs):
        return model_helpers.SpatialBN(self, *args, **kwargs)

    def InstanceNorm(self, *args, **kwargs):
        return model_helpers.InstanceNorm(self, *args,
                                          use_cudnn=self.use_cudnn, **kwargs)

    def MaxPool(self, *args, **kwargs):
        return model_helpers.MaxPool(self, *args, use_cudnn=self.use_cudnn,
                                     **kwargs)

    def AveragePool(self, *args, **kwargs):
        return model_helpers.AveragePool(self, *args, use_cudnn=self.use_cudnn,
                                         **kwargs)

    def Concat(self, blobs_in, blob_out, **kwargs):
        """Depth Concat."""
        return self.net.Concat(
            blobs_in,
            [blob_out, "_" + blob_out + "_concat_dims"],
            order=self.order,
            **kwargs
        )[0]

    def DepthConcat(self, blobs_in, blob_out, **kwargs):
        """The old depth concat function - we should move to use concat."""
        print("DepthConcat is deprecated. use Concat instead.")
        return self.Concat(blobs_in, blob_out, **kwargs)

    def PRelu(self, blob_in, blob_out, num_channels=1, slope_init=None,
              **kwargs):
        """PRelu"""
        slope_init = (
            slope_init if slope_init else ('ConstantFill', {'value': 0.25}))
        if self.init_params:
            slope = self.param_init_net.__getattr__(slope_init[0])(
                [],
                blob_out + '_slope',
                shape=[num_channels],
                **slope_init[1]
            )
        else:
            slope = core.ScopedBlobReference(
                blob_out + '_slope', self.param_init_net)

        self.params.extend([slope])

        return self.net.PRelu([blob_in, slope], [blob_out])

    def Relu(self, blob_in, blob_out, **kwargs):
        """Relu."""
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
        return self.net.Relu(blob_in, blob_out, order=self.order, **kwargs)

    def Transpose(self, blob_in, blob_out, **kwargs):
        """Transpose."""
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
        return self.net.Transpose(blob_in, blob_out, **kwargs)

    def Sum(self, blob_in, blob_out, **kwargs):
        """Sum"""
        return self.net.Sum(blob_in, blob_out, **kwargs)

    def Iter(self, blob_out, **kwargs):
        if 'device_option' in kwargs:
            del kwargs['device_option']
        self.param_init_net.ConstantFill(
            [], blob_out, shape=[1], value=0, dtype=core.DataType.INT64,
            device_option=core.DeviceOption(caffe2_pb2.CPU, 0),
            **kwargs)
        return self.net.Iter(blob_out, blob_out, **kwargs)

    def Accuracy(self, blob_in, blob_out, **kwargs):
        dev = kwargs['device_option'] if 'device_option' in kwargs \
            else scope.CurrentDeviceScope()
        is_cpu = dev is None or dev.device_type == caffe2_pb2.CPU

        # We support top_k > 1 only on CPU
        if not is_cpu and 'top_k' in kwargs and kwargs['top_k'] > 1:
            pred_host = self.net.CopyGPUToCPU(blob_in[0], blob_in[0] + "_host")
            label_host = self.net.CopyGPUToCPU(blob_in[1], blob_in[1] + "_host")

            # Now use the Host version of the accuracy op
            self.net.Accuracy([pred_host, label_host],
                              blob_out,
                              device_option=core.DeviceOption(caffe2_pb2.CPU, 0),
                              **kwargs)
        else:
            self.net.Accuracy(blob_in, blob_out)

    def PadImage(
        self, blob_in, blob_out, **kwargs
    ):
        self.net.PadImage(blob_in, blob_out, **kwargs)

    @property
    def XavierInit(self):
        return ('XavierFill', {})

    def ConstantInit(self, value):
        return ('ConstantFill', dict(value=value))

    @property
    def MSRAInit(self):
        return ('MSRAFill', {})

    @property
    def ZeroInit(self):
        return ('ConstantFill', {})

    def AddWeightDecay(self, weight_decay):
        """Adds a decay to weights in the model.

        This is a form of L2 regularization.

        Args:
            weight_decay: strength of the regularization
        """
        if weight_decay <= 0.0:
            return
        wd = self.param_init_net.ConstantFill([], 'wd', shape=[1],
                                              value=weight_decay)
        ONE = self.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
        for param in self.GetWeights():
            #  Equivalent to: grad += wd * param
            grad = self.param_to_grad[param]
            self.net.WeightedSum(
                [grad, ONE, param, wd],
                grad,
            )

    @property
    def CPU(self):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CPU
        return device_option

    @property
    def GPU(self, gpu_id=0):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.CUDA
        device_option.cuda_gpu_id = gpu_id
        return device_option

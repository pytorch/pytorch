## @package cnn
# Module caffe2.python.cnn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, scope
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

        self.weights = []
        self.biases = []
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

    def _FC_or_packed_FC(
        self, op_call, blob_in, blob_out, dim_in, dim_out, weight_init=None,
        bias_init=None, **kwargs
    ):
        """FC"""
        weight_init = weight_init or ('XavierFill', {})
        bias_init = bias_init or ('ConstantFill', {})
        blob_out = blob_out or self.net.NextName()
        if self.init_params:
            weight = self.param_init_net.__getattr__(weight_init[0])(
                [],
                blob_out + '_w',
                shape=[dim_out, dim_in],
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

        if 'freeze_bias' in kwargs:
            self.params.extend([weight])
        else:
            self.params.extend([weight, bias])

        self.weights.append(weight)
        self.biases.append(bias)
        return op_call([blob_in, weight, bias], blob_out, **kwargs)

    def FC(self, *args, **kwargs):
        return self._FC_or_packed_FC(self.net.FC, *args, **kwargs)

    def PackedFC(self, *args, **kwargs):
        return self._FC_or_packed_FC(self.net.PackedFC, *args, **kwargs)

    def FC_Decomp(
        self, blob_in, blob_out, dim_in, dim_out,
        rank_approx=5, weight_init=None,
        bias_init=None, **kwargs
    ):
        """FC_Decomp version
        Here we assume that the rank of original input is bigger than 5.
        """
        weight_init = weight_init if weight_init else ('XavierFill', {})
        bias_init = bias_init if bias_init else ('ConstantFill', {})
        blob_out = blob_out or self.net.NextName()
        u = self.param_init_net.__getattr__(weight_init[0])(
            [],
            blob_out + '_u',
            shape=[dim_out, rank_approx],
            **weight_init[1]
        )
        v = self.param_init_net.__getattr__(weight_init[0])(
            [],
            blob_out + '_v',
            shape=[dim_in, rank_approx],
            **weight_init[1]
        )
        bias = self.param_init_net.__getattr__(bias_init[0])(
            [],
            blob_out + '_b',
            shape=[dim_out, ],
            **bias_init[1]
        )
        self.params.extend([u, v, bias])
        return self.net.FC_Decomp([blob_in, u, v, bias], blob_out, **kwargs)

    def FC_Prune(
        self, blob_in, blob_out, dim_in, dim_out,
        weight_init=None, bias_init=None, mask_init=None,
        threshold=0.00001, need_compress_rate=False,
        comp_lb=0.05,
        **kwargs
    ):
        """FC_Prune version
        Runnable so far. Great!:)
        """
        weight_init = weight_init if weight_init else ('XavierFill', {})
        bias_init = bias_init if bias_init else ('ConstantFill', {})
        mask_init = mask_init if mask_init else ('ConstantFill', {})
        blob_out = blob_out or self.net.NextName()
        compress_rate = blob_out + '_compress_rate'
        if self.init_params:
            compress_lb = self.param_init_net.ConstantFill(
                [],
                blob_out + '_lb',
                shape=[1],
                value=comp_lb
            )
            weight = self.param_init_net.__getattr__(weight_init[0])(
                [],
                blob_out + '_w',
                shape=[dim_out, dim_in],
                **weight_init[1]
            )
            mask = self.param_init_net.ConstantFill(
                [],
                blob_out + '_m',
                shape=[dim_out, dim_in],
                value=1.0
            )
            ag_dw = self.param_init_net.__getattr__(mask_init[0])(
                [],
                blob_out + '_ag_dw',
                shape=[dim_out, dim_in],
                **mask_init[1]
            )
            bias = self.param_init_net.__getattr__(bias_init[0])(
                [],
                blob_out + '_b',
                shape=[dim_out, ],
                **bias_init[1]
            )
            mask_seq = self.param_init_net.__getattr__(mask_init[0])(
                [],
                blob_out + '_mask_seq',
                shape=[dim_out, dim_in],
                **mask_init[1]
            )
            thres = self.param_init_net.ConstantFill(
                [],
                blob_out + '_thres',
                shape=[1],
                value=threshold
            )
        else:
            compress_lb = core.ScopedBlobReference(
                blob_out + '_lb', self.param_init_net)
            weight = core.ScopedBlobReference(
                blob_out + '_w', self.param_init_net)
            bias = core.ScopedBlobReference(
                blob_out + '_b', self.param_init_net)
            mask = core.ScopedBlobReference(
                blob_out + '_m', self.param_init_net)
            ag_dw = core.ScopedBlobReference(
                blob_out + '_ag_dw', self.param_init_net)
            mask_seq = core.ScopedBlobReference(
                blob_out + '_mask_seq', self.param_init_net)
            thres = core.ScopedBlobReference(
                blob_out + '_thres', self.param_init_net)

        self.params.extend([weight, bias])
        if need_compress_rate:
            return self.net.FC_Prune([blob_in, weight, mask,
                                      bias, ag_dw, mask_seq,
                                      thres, compress_lb],
                                     [blob_out, compress_rate], **kwargs)
        else:
            return self.net.FC_Prune([blob_in, weight, mask,
                                      bias, ag_dw, mask_seq,
                                      thres, compress_lb],
                                     blob_out, **kwargs)

    def FC_Sparse(
        self, blob_in, blob_out, w_csr, iw, jw, bias,
        **kwargs
    ):
        """FC_Sparse: Only takes in alocated weights"""
        if not (w_csr and iw and jw and bias):
            print("Warning...")
        self.params.extend([w_csr, iw, jw, bias])
        return self.net.FC_Sparse([blob_in, w_csr, iw, jw, bias],
                                  blob_out, **kwargs)

    def LRN(self, blob_in, blob_out, **kwargs):
        """LRN"""
        return self.net.LRN(
            blob_in,
            [blob_out, "_" + blob_out + "_scale"],
            order=self.order,
            **kwargs
        )[0]

    def Dropout(self, blob_in, blob_out, **kwargs):
        """Dropout"""
        return self.net.Dropout(
            blob_in, [blob_out, "_" + blob_out + "_mask"], **kwargs
        )[0]

    def MaxPool(self, blob_in, blob_out, **kwargs):
        """Max pooling"""
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
        return self.net.MaxPool(blob_in, blob_out, order=self.order, **kwargs)

    def AveragePool(self, blob_in, blob_out, **kwargs):
        """Average pooling"""
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
        return self.net.AveragePool(
            blob_in,
            blob_out,
            order=self.order,
            **kwargs
        )

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

    def InstanceNorm(self, blob_in, blob_out, dim_in, **kwargs):
        blob_out = blob_out or self.net.NextName()
        # Input: input, scale, bias
        # Output: output, saved_mean, saved_inv_std
        # scale: initialize with ones
        # bias: initialize with zeros

        def init_blob(value, suffix):
            return self.param_init_net.ConstantFill(
                [], blob_out + "_" + suffix, shape=[dim_in], value=value)
        scale, bias = init_blob(1.0, "s"), init_blob(0.0, "b")

        self.params.extend([scale, bias])
        self.weights.append(scale)
        self.biases.append(bias)
        blob_outs = [blob_out, blob_out + "_sm", blob_out + "_siv"]
        if 'is_test' in kwargs and kwargs['is_test']:
            blob_outputs = self.net.InstanceNorm(
                [blob_in, scale, bias], [blob_out],
                order=self.order, **kwargs)
            return blob_outputs
        else:
            blob_outputs = self.net.InstanceNorm(
                [blob_in, scale, bias], blob_outs,
                order=self.order, **kwargs)
            # Return the output
            return blob_outputs[0]

    def SpatialBN(self, blob_in, blob_out, dim_in, **kwargs):
        blob_out = blob_out or self.net.NextName()
        # Input: input, scale, bias, est_mean, est_inv_var
        # Output: output, running_mean, running_inv_var, saved_mean,
        #         saved_inv_var
        # scale: initialize with ones
        # bias: initialize with zeros
        # est mean: zero
        # est var: ones

        def init_blob(value, suffix):
            return self.param_init_net.ConstantFill(
                [], blob_out + "_" + suffix, shape=[dim_in], value=value)

        if self.init_params:
            scale, bias = init_blob(1.0, "s"), init_blob(0.0, "b")
            running_mean = init_blob(0.0, "rm")
            running_inv_var = init_blob(1.0, "riv")
        else:
            scale = core.ScopedBlobReference(
                    blob_out + '_s', self.param_init_net)
            bias = core.ScopedBlobReference(
                    blob_out + '_b', self.param_init_net)
            running_mean = core.ScopedBlobReference(
                    blob_out + '_rm', self.param_init_net)
            running_inv_var = core.ScopedBlobReference(
                    blob_out + '_riv', self.param_init_net)

        self.params.extend([scale, bias])
        self.computed_params.extend([running_mean, running_inv_var])
        self.weights.append(scale)
        self.biases.append(bias)
        blob_outs = [blob_out, running_mean, running_inv_var,
                     blob_out + "_sm", blob_out + "_siv"]
        if 'is_test' in kwargs and kwargs['is_test']:
            blob_outputs = self.net.SpatialBN(
                [blob_in, scale, bias, blob_outs[1], blob_outs[2]], [blob_out],
                order=self.order, **kwargs)
            return blob_outputs
        else:
            blob_outputs = self.net.SpatialBN(
                [blob_in, scale, bias, blob_outs[1], blob_outs[2]], blob_outs,
                order=self.order, **kwargs)
            # Return the output
            return blob_outputs[0]

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

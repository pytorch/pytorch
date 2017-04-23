## @package cnn
# Module caffe2.python.cnn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import scope, brew
from caffe2.python.model_helper import ModelHelper
from caffe2.proto import caffe2_pb2


class CNNModelHelper(ModelHelper):
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

    def PadImage(self, blob_in, blob_out, **kwargs):
        self.net.PadImage(blob_in, blob_out, **kwargs)

    def ConvNd(self, *args, **kwargs):
        return brew.conv_nd(
            self,
            *args,
            use_cudnn=self.use_cudnn,
            order=self.order,
            cudnn_exhaustive_search=self.cudnn_exhaustive_search,
            ws_nbytes_limit=self.ws_nbytes_limit,
            **kwargs
        )

    def Conv(self, *args, **kwargs):
        return brew.conv(
            self,
            *args,
            use_cudnn=self.use_cudnn,
            order=self.order,
            cudnn_exhaustive_search=self.cudnn_exhaustive_search,
            ws_nbytes_limit=self.ws_nbytes_limit,
            **kwargs
        )

    def ConvTranspose(self, *args, **kwargs):
        return brew.conv_transpose(
            self,
            *args,
            use_cudnn=self.use_cudnn,
            order=self.order,
            cudnn_exhaustive_search=self.cudnn_exhaustive_search,
            ws_nbytes_limit=self.ws_nbytes_limit,
            **kwargs
        )

    def GroupConv(self, *args, **kwargs):
        return brew.group_conv(
            self,
            *args,
            use_cudnn=self.use_cudnn,
            order=self.order,
            cudnn_exhaustive_search=self.cudnn_exhaustive_search,
            ws_nbytes_limit=self.ws_nbytes_limit,
            **kwargs
        )

    def GroupConv_Deprecated(self, *args, **kwargs):
        return brew.group_conv_deprecated(
            self,
            *args,
            use_cudnn=self.use_cudnn,
            order=self.order,
            cudnn_exhaustive_search=self.cudnn_exhaustive_search,
            ws_nbytes_limit=self.ws_nbytes_limit,
            **kwargs
        )

    def FC(self, *args, **kwargs):
        return brew.fc(self, *args, **kwargs)

    def PackedFC(self, *args, **kwargs):
        return brew.packed_fc(self, *args, **kwargs)

    def FC_Prune(self, *args, **kwargs):
        return brew.fc_prune(self, *args, **kwargs)

    def FC_Decomp(self, *args, **kwargs):
        return brew.fc_decomp(self, *args, **kwargs)

    def FC_Sparse(self, *args, **kwargs):
        return brew.fc_sparse(self, *args, **kwargs)

    def Dropout(self, *args, **kwargs):
        return brew.dropout(self, *args, **kwargs)

    def LRN(self, *args, **kwargs):
        return brew.lrn(self, *args, **kwargs)

    def Softmax(self, *args, **kwargs):
        return brew.softmax(self, *args, use_cudnn=self.use_cudnn,
                                     **kwargs)

    def SpatialBN(self, *args, **kwargs):
        return brew.spatial_bn(self, *args, order=self.order, **kwargs)

    def InstanceNorm(self, *args, **kwargs):
        return brew.instance_norm(self, *args, order=self.order,
                                          **kwargs)

    def Relu(self, *args, **kwargs):
        return brew.relu(self, *args, order=self.order,
                                  use_cudnn=self.use_cudnn, **kwargs)

    def PRelu(self, *args, **kwargs):
        return brew.prelu(self, *args, **kwargs)

    def Concat(self, *args, **kwargs):
        return brew.concat(self, *args, order=self.order, **kwargs)

    def DepthConcat(self, *args, **kwargs):
        """The old depth concat function - we should move to use concat."""
        print("DepthConcat is deprecated. use Concat instead.")
        return self.Concat(*args, **kwargs)

    def Sum(self, *args, **kwargs):
        return brew.sum(self, *args, **kwargs)

    def Transpose(self, *args, **kwargs):
        return brew.transpose(self, *args, use_cudnn=self.use_cudnn,
                                       **kwargs)

    def Iter(self, *args, **kwargs):
        return brew.iter(self, *args, **kwargs)

    def Accuracy(self, *args, **kwargs):
        return brew.accuracy(self, *args, **kwargs)

    def MaxPool(self, *args, **kwargs):
        return brew.max_pool(self, *args, use_cudnn=self.use_cudnn,
                                     order=self.order, **kwargs)

    def AveragePool(self, *args, **kwargs):
        return brew.average_pool(self, *args, use_cudnn=self.use_cudnn,
                                         order=self.order, **kwargs)

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

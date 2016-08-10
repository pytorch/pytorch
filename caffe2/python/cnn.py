from caffe2.python import core
from caffe2.proto import caffe2_pb2

import logging


class CNNModelHelper(object):
    """A helper model so we can write CNN models more easily, without having to
    manually define parameter initializations and operators separately.
    """

    def __init__(self, order="NCHW", name=None,
                 use_cudnn=True, cudnn_exhaustive_search=False,
                 ws_nbytes_limit=None, init_params=True):
        if name is None:
            name = "CNN"
        self.net = core.Net(name)
        self.param_init_net = core.Net(name + '_init')
        self.params = []
        self.param_to_grad = {}
        self.weights = []
        self.biases = []
        self.order = order
        self.use_cudnn = use_cudnn
        self.cudnn_exhaustive_search = cudnn_exhaustive_search
        self.ws_nbytes_limit = ws_nbytes_limit
        self.init_params = init_params
        self.gradient_ops_added = False
        if self.order != "NHWC" and self.order != "NCHW":
            raise ValueError(
                "Cannot understand the CNN storage order %s." % self.order
            )

    def Proto(self):
        return self.net.Proto()

    def RunAllOnGPU(self, *args, **kwargs):
        self.param_init_net.RunAllOnGPU(*args, **kwargs)
        self.net.RunAllOnGPU(*args, **kwargs)

    def CreateDB(self, blob_out, db, db_type, **kwargs):
        dbreader = self.param_init_net.CreateDB(
            [], blob_out, db=db, db_type=db_type, **kwargs)
        return dbreader

    def ImageInput(
            self, blob_in, blob_out, **kwargs
    ):
        assert len(blob_in) == 1
        assert len(blob_out) == 2
        """Image Input."""
        if self.order == "NCHW":
            data, label = self.net.ImageInput(
                blob_in, [blob_out[0] + '_nhwc', blob_out[1]], **kwargs)
            data = self.net.NHWC2NCHW(data, blob_out[0])
        else:
            data, label = self.net.ImageInput(
                blob_in, blob_out, **kwargs)
        return data, label

    def TensorProtosDBInput(
        self, unused_blob_in, blob_out, batch_size, db, db_type, **kwargs
    ):
        """TensorProtosDBInput."""
        dbreader_name = "dbreader_" + db
        dbreader = self.param_init_net.CreateDB(
            [], dbreader_name,
            db=db, db_type=db_type)
        return self.net.TensorProtosDBInput(
            dbreader, blob_out, batch_size=batch_size)

    def Conv(
        self, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None,
        bias_init=None, **kwargs
    ):
        """Convolution. We intentionally do not provide odd kernel/stride/pad
        settings in order to discourage the use of odd cases.
        """
        weight_init = weight_init if weight_init else ('XavierFill', {})
        bias_init = bias_init if bias_init else ('ConstantFill', {})
        blob_out = blob_out or self.net.NextName()
        weight_shape = (
            [dim_out, dim_in, kernel, kernel]
            if self.order == "NCHW" else [dim_out, kernel, kernel, dim_in]
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
        return self.net.Conv(
            [blob_in, weight, bias],
            blob_out,
            kernel=kernel,
            order=self.order,
            **kwargs
        )

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
        weight_init,
        bias_init,
        group=1,
        **kwargs
    ):
        """Convolution. We intentionally do not provide odd kernel/stride/pad
        settings in order to discourage the use of odd cases.
        """
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit
        if dim_in % group:
            raise ValueError("dim_in should be divisible by group.")
        splitted_blobs = self.net.DepthSplit(
            blob_in,
            ['_' + blob_out + '_gconv_split_' + str(i) for i in range(group)],
            dimensions=[dim_in / group for i in range(group)],
            order=self.order
        )
        weight_shape = (
            [dim_out / group, dim_in / group, kernel, kernel]
            if self.order == "NCHW" else
            [dim_out / group, kernel, kernel, dim_in / group]
        )
        conv_blobs = []
        for i in range(group):
            if self.init_params:
                weight = self.param_init_net.__getattr__(weight_init[0])(
                    [],
                    blob_out + '_gconv_%d_w' % i,
                    shape=weight_shape,
                    **weight_init[1]
                )
                bias = self.param_init_net.__getattr__(bias_init[0])(
                    [],
                    blob_out + '_gconv_%d_b' % i,
                    shape=[dim_out / group],
                    **bias_init[1]
                )
            else:
                weight = core.ScopedBlobReference(
                    blob_out + '_gconv_%d_w' % i, self.param_init_net)
                bias = core.ScopedBlobReference(
                    blob_out + '_gconv_%d_b' % i, self.param_init_net)
            self.params.extend([weight, bias])
            self.weights.append(weight)
            self.biases.append(bias)
            conv_blobs.append(
                splitted_blobs[i].Conv(
                    [weight, bias],
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

    def FC(
        self, blob_in, blob_out, dim_in, dim_out, weight_init=None,
        bias_init=None, **kwargs
    ):
        """FC"""
        weight_init = weight_init if weight_init else ('XavierFill', {})
        bias_init = bias_init if bias_init else ('ConstantFill', {})
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
        self.params.extend([weight, bias])
        return self.net.FC([blob_in, weight, bias], blob_out, **kwargs)

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
            [blob_out, "_" + blob_out + "_condat_dims"],
            order=self.order,
            **kwargs
        )[0]

    def DepthConcat(self, blobs_in, blob_out, **kwargs):
        """The old depth concat function - we should move to use concat."""
        print("DepthConcat is deprecated. use Concat instead.")
        return self.Concat(blobs_in, blob_out, **kwargs)

    def Relu(self, blob_in, blob_out, **kwargs):
        """Relu."""
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
        return self.net.Relu(blob_in, blob_out, order=self.order, **kwargs)

    def Transpose(self, blob_in, blob_out, **kwargs):
        """Transpose."""
        return self.net.Transpose(blob_in, blob_out, **kwargs)

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
        scale, bias = init_blob(1.0, "s"), init_blob(0.0, "b")
        self.params.extend([scale, bias])
        self.weights.append(scale)
        self.biases.append(bias)
        blob_outs = [blob_out, blob_out + "_rm", blob_out + "_riv",
                     blob_out + "_sm", blob_out + "_siv"]
        blob_outputs = self.net.SpatialBN(
            [blob_in, scale, bias, blob_outs[1], blob_outs[2]], blob_outs,
            order=self.order, **kwargs)
        # Return the output
        return blob_outputs[0]

    def Iter(self, blob_out, **kwargs):
        if 'device_option' in kwargs:
            del kwargs['device_option']
        self.param_init_net.ConstantFill(
            [], blob_out, shape=[1], value=0, dtype=core.DataType.INT32,
            device_option=core.DeviceOption(caffe2_pb2.CPU, 0),
            **kwargs)
        return self.net.Iter(blob_out, blob_out, **kwargs)

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

    def AddGradientOperators(self, *args, **kwargs):
        if self.gradient_ops_added:
            raise RuntimeError("You cannot run AddGradientOperators twice.")
        self.gradient_ops_added = True
        grad_map = self.net.AddGradientOperators(*args, **kwargs)
        for p in self.params:
            if str(p) in grad_map:
                self.param_to_grad[p] = grad_map[str(p)]
        return grad_map

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

    def __getattr__(self, op_type):
        """Catch-all for all other operators, mostly those without params."""
        if not core.IsOperator(op_type):
            raise RuntimeError(
                'Method ' + op_type + ' is not a registered operator.'
            )
        # known_working_ops are operators that do not need special care.
        known_working_ops = [
            "Accuracy",
            "AveragedLoss",
            "Cast",
            "LabelCrossEntropy",
            "LearningRate",
            "Print",
            "Scale",
            "Snapshot",
            "Softmax",
            "StopGradient",
            "Summarize",
            "WeightedSum",
        ]
        if op_type not in known_working_ops:
            logging.warning("You are creating an op that the CNNModelHelper "
                            "does not recognize: {}.".format(op_type))
        return self.net.__getattr__(op_type)

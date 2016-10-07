from caffe2.python import core
from caffe2.python.model_helper import ModelHelperBase
from caffe2.proto import caffe2_pb2

import logging


class CNNModelHelper(ModelHelperBase):
    """A helper model so we can write CNN models more easily, without having to
    manually define parameter initializations and operators separately.
    """

    def __init__(self, order="NCHW", name=None,
                 use_cudnn=True, cudnn_exhaustive_search=False,
                 ws_nbytes_limit=None, init_params=True):
        super(CNNModelHelper, self).__init__(
            name="CNN" if name is None else name, init_params=init_params)

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

    def ImageInput(
            self, blob_in, blob_out, **kwargs
    ):
        """Image Input."""
        if self.order == "NCHW":
            data, label = self.net.ImageInput(
                blob_in, [blob_out[0] + '_nhwc', blob_out[1]], **kwargs)
            data = self.net.NHWC2NCHW(data, blob_out[0])
        else:
            data, label = self.net.ImageInput(
                blob_in, blob_out, **kwargs)
        return data, label

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

    def _FC_or_packed_FC(
        self, op_call, blob_in, blob_out, dim_in, dim_out, weight_init=None,
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

    def Relu(self, blob_in, blob_out, **kwargs):
        """Relu."""
        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
        return self.net.Relu(blob_in, blob_out, order=self.order, **kwargs)

    def Transpose(self, blob_in, blob_out, **kwargs):
        """Transpose."""
        return self.net.Transpose(blob_in, blob_out, **kwargs)

    def Sum(self, blob_in, blob_out, **kwargs):
        """Sum"""
        return self.net.Sum(blob_in, blob_out, **kwargs)

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
        running_mean = init_blob(0.0, "rm")
        running_inv_var = init_blob(1.0, "riv")
        self.params.extend([scale, bias])
        self.weights.append(scale)
        self.biases.append(bias)
        blob_outs = [blob_out, running_mean, running_inv_var,
                     blob_out + "_sm", blob_out + "_siv"]
        blob_outputs = self.net.SpatialBN(
            [blob_in, scale, bias, running_mean, running_inv_var], blob_outs,
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
        for param in self.weights:
            #  Equivalent to: grad += wd * param
            self.net.WeightedSum([self.param_to_grad[param], ONE, param, wd])

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

    def LSTM(self, input_blob, seq_lengths, initial_states, dim_in, dim_out,
             scope=None):
        def s(name):
            # We have to manually scope due to our internal/external blob
            # relationships.
            scope_name = scope or str(input_blob)
            return "{}/{}".format(str(scope_name), str(name))

        (hidden_input_blob, cell_input_blob) = initial_states

        input_blob = self.FC(input_blob, s("i2h"),
                             dim_in=dim_in, dim_out=4 * dim_out, axis=2)

        step_net = CNNModelHelper(name="LSTM")
        step_net.Proto().external_input.extend([
            str(seq_lengths),
            "input_t",
            "timestep",
            "hidden_t_prev",
            "cell_t_prev",
            s("gates_t_w"),
            s("gates_t_b"),
        ])
        step_net.Proto().type = "simple"
        step_net.Proto().external_output.extend(
            ["hidden_t", "cell_t", s("gates_t")])
        step_net.FC("hidden_t_prev", s("gates_t"),
                    dim_in=dim_out, dim_out=4 * dim_out, axis=2)
        step_net.net.Sum([s("gates_t"), "input_t"], [s("gates_t")])
        step_net.net.LSTMUnit(
            ["cell_t_prev", s("gates_t"), str(seq_lengths), "timestep"],
            ["hidden_t", "cell_t"])

        links = [
            ("hidden_t_prev", s("hidden"), 0),
            ("hidden_t", s("hidden"), 1),
            ("cell_t_prev", s("cell"), 0),
            ("cell_t", s("cell"), 1),
            (s("gates_t"), s("gates"), 0),
            ("input_t", str(input_blob), 0),
        ]
        link_internal, link_external, link_offset = zip(*links)

        # # Initialize params for step net in the parent net
        # for op in step_net.param_init_net.Proto().op:

        # Set up the backward links
        backward_ops, backward_mapping = core.GradientRegistry.GetBackwardPass(
            step_net.Proto().op,
            {"hidden_t": "hidden_t_grad", "cell_t": "cell_t_grad"})
        backward_mapping = {str(k): str(v) for k, v
                            in backward_mapping.items()}
        backward_step_net = core.Net("LSTMBackward")
        del backward_step_net.Proto().op[:]
        backward_step_net.Proto().op.extend(backward_ops)

        backward_links = [
            ("hidden_t_prev_grad", s("hidden_grad"), 0),
            ("hidden_t_grad", s("hidden_grad"), 1),
            ("cell_t_prev_grad", s("cell_grad"), 0),
            ("cell_t_grad", s("cell_grad"), 1),
            (s("gates_t_grad"), s("gates_grad"), 0),
        ]

        backward_link_internal, backward_link_external, \
            backward_link_offset = zip(*backward_links)

        backward_step_net.Proto().external_input.extend(
            ["hidden_t_grad", "cell_t_grad"])
        backward_step_net.Proto().external_input.extend(
            step_net.Proto().external_input)
        backward_step_net.Proto().external_input.extend(
            step_net.Proto().external_output)

        output, _, _, hidden_state, cell_state = self.net.RecurrentNetwork(
            [input_blob, seq_lengths,
                s("gates_t_w"), s("gates_t_b"),
                hidden_input_blob, cell_input_blob],
            [s("output"), s("hidden"), s("cell"),
                s("hidden_output"), s("cell_output")],
            param=[str(p) for p in step_net.params],
            param_gradient=[backward_mapping[str(p)] for p in step_net.params],
            alias_src=[s("hidden"), s("hidden"), s("cell")],
            alias_dst=[s("output"), s("hidden_output"), s("cell_output")],
            alias_offset=[1, -1, -1],
            recurrent_states=[s("hidden"), s("cell")],
            recurrent_inputs=[str(hidden_input_blob), str(cell_input_blob)],
            recurrent_sizes=[dim_out, dim_out],
            link_internal=link_internal,
            link_external=link_external,
            link_offset=link_offset,
            backward_link_internal=backward_link_internal,
            backward_link_external=backward_link_external,
            backward_link_offset=backward_link_offset,
            backward_alias_src=[s("gates_grad")],
            backward_alias_dst=[str(input_blob) + "_grad"],
            backward_alias_offset=[0],
            scratch=[s("gates")],
            backward_scratch=[s("gates_grad")],
            scratch_sizes=[4 * dim_out],
            step_net=str(step_net.Proto()),
            backward_step_net=str(backward_step_net.Proto()),
            timestep="timestep")
        self.param_init_net.Proto().op.extend(
            step_net.param_init_net.Proto().op)
        self.params += step_net.params
        for p in step_net.params:
            if str(p) in backward_mapping:
                self.param_to_grad[p] = backward_mapping[str(p)]
        self.weights += step_net.weights
        self.biases += step_net.biases
        return output, hidden_state, cell_state

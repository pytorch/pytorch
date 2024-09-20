# mypy: allow-untyped-defs
import torch


class OnednnLinear(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype):
        super().__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))
        if dense_module.bias is not None:
            # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
            # we use fp32 dtype.
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

    @torch.jit.script_method
    def forward(self, x):
        x_mkldnn = x if x.is_onednn else x.to_mkldnn()
        y_mkldnn = torch._C._nn.onednn_linear(x_mkldnn, self.weight, self.bias)
        y = y_mkldnn if x.is_onednn else y_mkldnn.to_dense()
        return y


class _OnednnConvNd(torch.jit.ScriptModule):
    """Common base of OnednnConv1d and OnednnConv2d."""

    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module):
        super().__init__()

        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups

        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
            # we use fp32 dtype.
            # TODO: Remove this once ScriptModule supports registering None buffer
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def forward(self, x):
        return torch.mkldnn_convolution(
            x,
            self.weight,
            self.bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups)


class OnednnConv1d(_OnednnConvNd):
    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)

        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]


class OnednnConv2d(_OnednnConvNd):
    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)

        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

class OnednnConv3d(_OnednnConvNd):
    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)

        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]


class OnednnBatchNorm(torch.jit.ScriptModule):
    __constants__ = ['exponential_average_factor', 'eps']

    def __init__(self, dense_module):
        super().__init__()

        assert not dense_module.training
        assert dense_module.track_running_stats
        assert dense_module.affine

        if dense_module.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = dense_module.momentum
        self.eps = dense_module.eps

        self.register_buffer('weight', dense_module.weight.to_mkldnn())
        self.register_buffer('bias', dense_module.bias.to_mkldnn())
        self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
        self.register_buffer('running_var', dense_module.running_var.to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        weight = self.weight.to_dense()
        bias = self.bias.to_dense()
        running_mean = self.running_mean.to_dense()
        running_var = self.running_var.to_dense()
        return (weight, bias, running_mean, running_var, self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.running_mean = state[2].to_mkldnn()
        self.running_var = state[3].to_mkldnn()
        self.training = state[4]

    @torch.jit.script_method
    def forward(self, x):
        return torch.batch_norm(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            False,  # training
            self.exponential_average_factor,
            self.eps,
            False,  # cuda_enabled
        )

class OnednnPrelu(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype):
        super().__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.training = state[1]

    @torch.jit.script_method
    def forward(self, x):
        x_mkldnn = x if x.is_onednn else x.to_mkldnn()
        y_mkldnn = torch.prelu(x_mkldnn, self.weight)
        y = y_mkldnn if x.is_onednn else y_mkldnn.to_dense()
        return y

def to_mkldnn(module, dtype=torch.float):
    assert dtype in [torch.float, torch.bfloat16, torch.half], \
        "ONEDNN only support float, bfloat16, and half path now"

    def m_fn(m, d):
        if isinstance(m, torch.nn.Linear):
            return OnednnLinear(m, d)
        elif isinstance(m, torch.nn.Conv1d):
            return OnednnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return OnednnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return OnednnConv3d(m, d)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            # For batchnorm bf16 path, OneDNN requires weight and bias need fp32 dtype.
            # so it doesn't need dtype argument.
            return OnednnBatchNorm(m)
        elif isinstance(m, torch.nn.PReLU):
            return OnednnPrelu(m, d)
        else:
            return m

    def m_fn_rec(m, d):
        new_m = m_fn(m, d)
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        return new_m

    return m_fn_rec(module, dtype)

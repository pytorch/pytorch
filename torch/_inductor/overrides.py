import copy
import itertools
import logging
import operator
import random
import weakref

import torch
import torch.nn as nn
from torch import _prims
from torch.fx.experimental.optimization import (
    matches_module_pattern,
    replace_node_module,
)
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.overrides import TorchFunctionMode

log = logging.getLogger(__name__)


class AutogradMonkeypatch(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        if func is replacements:
            return replacements[func](*args, **kwargs)
        return func(*args, **kwargs)


patch_functions = AutogradMonkeypatch


def replace_fx(gm: torch.fx.GraphModule):
    # Sometimes patch_functions() misses things already in the graph
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and node.target in replacements:
            with gm.graph.inserting_before(node):
                node.replace_all_uses_with(
                    gm.graph.call_function(
                        replacements[node.target], node.args, node.kwargs
                    )
                )
            gm.graph.erase_node(node)
    gm.recompile()
    return gm


class UnaryFusionOp:
    def __init__(self, post_op, scalars=None, algorithm=None):
        self.post_op = post_op
        self.scalars = scalars if scalars else []
        self.algorithm = algorithm if algorithm else ""


class ConvUnary2d(nn.Conv2d):
    def __init__(
        self,
        conv,
        unary,
        op_name,
        op_info,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        device,
        dtype,
    ):
        super(ConvUnary2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._update_module_params(conv, unary, op_name, op_info)

    def _update_module_params(self, conv, unary, op_name, op_info):
        self.__dict__ = copy.deepcopy(conv.__dict__)

        self.attr = op_name

        assert all(hasattr(unary, item) for item in op_info.scalars)
        self.scalars = [getattr(unary, item) for item in op_info.scalars]

        algorithm = ""
        if op_info.algorithm:
            assert hasattr(unary, op_info.algorithm)
            algorithm = getattr(unary, op_info.algorithm)
        self.algorithm = algorithm

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
                self.attr,
                self.scalars,
                self.algorithm,
            )
        return torch.ops.mkldnn._convolution_pointwise(
            input,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.attr,
            self.scalars,
            self.algorithm,
        )

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)


class ConvBinary2d(nn.Conv2d):
    def __init__(
        self,
        conv,
        op_name,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        device,
        dtype,
    ):
        super(ConvBinary2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._update_module_params(conv, op_name)

    def _update_module_params(self, conv, op_name):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.attr = op_name

    def _conv_forward(self, input, other, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                other,
                weight,
                bias,
                _pair(0),
                self.stride,
                self.dilation,
                self.groups,
                self.attr,
            )
        return torch.ops.mkldnn._convolution_pointwise(
            input,
            other,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.attr,
        )

    def forward(self, input, other):
        return self._conv_forward(input, other, self.weight, self.bias)


class LinearUnary(nn.Linear):
    def __init__(
        self,
        linear,
        eltwise,
        op_name,
        op_info,
        in_features,
        out_features,
        bias,
        device,
        dtype,
    ):
        super(LinearUnary, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._update_module_params(linear, eltwise, op_name, op_info)

    def _update_module_params(self, linear, eltwise, op_name, op_info):
        self.__dict__ = copy.deepcopy(linear.__dict__)

        self.attr = op_name

        assert all(hasattr(eltwise, item) for item in op_info.scalars)
        self.scalars = [getattr(eltwise, item) for item in op_info.scalars]

        algorithm = ""
        if op_info.algorithm:
            assert hasattr(eltwise, op_info.algorithm)
            algorithm = getattr(eltwise, op_info.algorithm)
        self.algorithm = algorithm

    def forward(self, input):
        y = torch.ops.mkldnn._linear_pointwise(
            input, self.weight, self.bias, self.attr, self.scalars, self.algorithm
        )
        return y


class LinearBinary(nn.Linear):
    def __init__(self, linear, in_features, out_features, bias, device, dtype, attr):
        super(LinearBinary, self).__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._update_module_params(linear, attr)

    def _update_module_params(self, linear, attr):
        self.__dict__ = copy.deepcopy(linear.__dict__)

        self.attr = attr

    def forward(self, input, other):
        y = torch.ops.mkldnn._linear_pointwise(
            input, other, self.weight, self.bias, self.attr
        )
        return y


def fuse_conv_unary_eval(conv, unary, op_name, op_info):
    assert not (conv.training), "Fusion only for eval!"
    return ConvUnary2d(
        conv,
        unary,
        op_name,
        op_info,
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
        conv.padding_mode,
        conv.weight.device,
        conv.weight.dtype,
    )


def fuse_conv_binary_eval(conv, op_name):
    assert not (conv.training), "Fusion only for eval!"
    return ConvBinary2d(
        conv,
        op_name,
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
        conv.padding_mode,
        conv.weight.device,
        conv.weight.dtype,
    )


def is_bfloat16_module(m):
    weight_is_bf16 = m.weight.dtype == torch.bfloat16
    bias_is_bf16 = m.bias is None or m.bias.dtype == torch.bfloat16
    return weight_is_bf16 and bias_is_bf16


def bf16_only_node(m):
    if type(m) in [nn.Linear]:
        return True
    else:
        return False


def fuse_linear_unary_eval(linear, eltwise, op_name, op_info):
    return LinearUnary(
        linear,
        eltwise,
        op_name,
        op_info,
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        linear.weight.device,
        linear.weight.dtype,
    )


def fuse_linear_binary_eval(linear, attr):
    assert not (linear.training), "Fusion only for eval!"
    linear_binary = LinearBinary(
        linear,
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        linear.weight.device,
        linear.weight.dtype,
        attr,
    )
    return linear_binary


def check_node_kind(current_node, modules, node_kind):
    if not isinstance(current_node, torch.fx.Node):
        return False
    if current_node.op != "call_module":
        return False
    if not isinstance(current_node.target, str):
        return False
    if current_node.target not in modules:
        return False
    if type(modules[current_node.target]) is not node_kind:
        return False
    return True


def check_node_is_binary(node):
    if (
        (node.op == "call_function" and node.target in [torch.add, torch.sub])
        or (node.op == "call_function" and node.target in [operator.add, operator.sub])
        or (
            node.op == "call_method"
            and node.target in [torch.Tensor.add, torch.Tensor.sub]
        )
    ):
        return True
    return False


def fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
    gm = fuse_unary(gm, example_inputs)
    gm = fuse_binary(gm)

    return gm


def fuse_unary(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu") for example_input in example_inputs
    )
    if not is_cpu:
        return gm
    modules = dict(gm.named_modules())

    for (pointwise_name, pointwise_info), (
        computation_name,
        fuse_func,
    ) in itertools.product(pointwise_op_map.items(), computation_op_map.items()):
        pattern = (computation_name, pointwise_info.post_op)
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if (
                    len(node.args[0].users) > 1
                ):  # Output of convolution is used by other nodes
                    continue
                computation_node = modules[node.args[0].target]
                unary = modules[node.target]
                eval_mode = all(not n.training for n in [computation_node, unary])
                if not eval_mode:
                    continue
                # only fuse for linear when the dtype is bf16
                if bf16_only_node(computation_node) and not is_bfloat16_module(
                    computation_node
                ):
                    continue
                fused_module = fuse_func(
                    computation_node, unary, pointwise_name, pointwise_info
                )
                replace_node_module(node.args[0], modules, fused_module)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                gm.graph.lint()
    gm.recompile()
    return gm


def _philox_rand_like_meta(input, seed, offset):
    return _prims.TensorMeta(input)


def _philox_rand_like(input, seed, offset):
    # placeholder only used in tracing
    return torch.rand_like(input)


def replace_and_fuse_for_binary(
    computation_node, node, fuse_func, attr, modules, index_node, index_pointwise
):
    fused_module = fuse_func(computation_node, attr)
    replace_node_module(node.args[index_node], modules, fused_module)
    node.args[index_node].args = node.args[index_node].args + (
        node.args[index_pointwise],
    )
    node.replace_all_uses_with(node.args[index_node])


def fuse_binary(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if check_node_is_binary(node) and (
            len(node.kwargs) != 2 or node.kwargs["alpha"] == 1.0
        ):
            for node_kind, fuse_func in computation_op_binary_op_fusion_map.items():
                if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
                    node.args[1], torch.fx.Node
                ):
                    continue
                tensor0_meta = node.args[0].meta.get("tensor_meta")
                tensor1_meta = node.args[1].meta.get("tensor_meta")
                if not tensor0_meta or not tensor1_meta:
                    continue
                if (
                    tensor0_meta.shape != tensor1_meta.shape
                    or tensor0_meta.stride != tensor1_meta.stride
                    or tensor0_meta.dtype != tensor1_meta.dtype
                ):
                    continue
                attr = binary_attr[node.target]
                index_list = supported_index_list[attr]
                for index_dict in index_list:
                    index_node = index_dict["index_computation"]
                    index_pointwise = index_dict["index_pointwise"]
                    if check_node_kind(node.args[index_node], modules, node_kind):
                        if len(node.args[index_node].users) > 1:
                            continue
                        computation_node = modules[node.args[index_node].target]
                        # only fuse for linear when the dtype is bf16
                        if bf16_only_node(computation_node) and not is_bfloat16_module(
                            computation_node
                        ):
                            continue
                        replace_and_fuse_for_binary(
                            computation_node,
                            node,
                            fuse_func,
                            attr,
                            modules,
                            index_node,
                            index_pointwise,
                        )
                        # Make sure the fused node is post node of node's inputs nodes.
                        node.append(node.args[index_node])
                        gm.graph.erase_node(node)
                        gm.graph.lint()
                        break

    gm.recompile()
    return gm


philox_rand_like = _prims._make_prim(
    schema="philox_rand_like(Tensor input, Tensor seed, int offset) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_rand_like_meta,
    impl_aten=_philox_rand_like,
    doc="",
)


def _philox_seed_like_meta(x):
    return _prims.TensorMeta(_philox_seed_like(x))


def _philox_seed_like(x):
    # we need a tensor input here so AOT autograd properly captures this
    # with just a device input, this becomes a constant
    return torch.tensor(random.randrange(2**31), device=x.device, dtype=torch.int32)


philox_seed_like = _prims._make_prim(
    schema="philox_seed_like(Tensor other) -> Tensor",
    return_type=_prims.RETURN_TYPE.NEW,
    meta=_philox_seed_like_meta,
    impl_aten=_philox_seed_like,
    doc="",
)


def null_ref():
    return None


class PhiloxRandomState:
    next_offset = 0
    seed = {}
    last_tracer_ref = null_ref

    @classmethod
    def reset(cls, tracer=None):
        cls.next_offset = 0
        cls.seed = {}
        cls.last_tracer_ref = weakref.ref(tracer) if tracer is not None else null_ref

    @classmethod
    def get_seed_offset(cls, x):
        modes = torch.fx.experimental.proxy_tensor.get_torch_dispatch_modes()
        proxy_modes = [m for m in modes if isinstance(m, ProxyTorchDispatchMode)]
        if proxy_modes:
            tracer = proxy_modes[0].tracer
            if cls.last_tracer_ref() is not tracer:
                # tracer changed, need to reset state
                cls.reset(tracer)
        else:
            # no tracer, need to reset state
            cls.reset()

        device = x.device
        if device not in cls.seed:
            # Compute the seed just once per trace so that we pass fewer
            # things from forward to backward
            cls.seed[device] = philox_seed_like(x)

        seed = cls.seed[device]
        offset = cls.next_offset
        cls.next_offset += x.numel()
        return seed, offset


class LowmemDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        scale = float(1.0 / (1.0 - p))
        seed, offset = PhiloxRandomState.get_seed_offset(x)
        ctx.save_for_backward(seed)
        ctx.offset = offset
        bool_mask = philox_rand_like(x, seed, offset) > p
        return bool_mask.to(x.dtype) * x * scale

    @staticmethod
    def backward(ctx, grad_output):
        p = ctx.p
        scale = float(1.0 / (1.0 - p))
        (seed,) = ctx.saved_tensors
        bool_mask = philox_rand_like(grad_output, seed, ctx.offset) > p
        return bool_mask.to(grad_output.dtype) * grad_output * scale, None


@torch.fx.wrap
def lowmem_dropout(input, p, training=True, inplace=False):
    if isinstance(input, torch.fx.Proxy):
        # double check we don't FX trace this
        return input.tracer.create_proxy(
            "call_function",
            lowmem_dropout,
            (input, p, training),
            {},
        )
    if not training or p == 0:
        return input
    result = LowmemDropout.apply(input, p)
    if inplace:
        input.copy_(result)
    return result


@torch.fx.wrap
def rand_like(x, **kwargs):
    if isinstance(x, torch.fx.Proxy):
        # double check we don't FX trace this
        return x.tracer.create_proxy("call_function", rand_like, (x), kwargs)
    assert kwargs.get("device", x.device) == x.device
    seed, offset = PhiloxRandomState.get_seed_offset(x)
    return philox_rand_like(x, seed, offset).to(kwargs.get("dtype", torch.float32))


replacements = {torch.nn.functional.dropout: lowmem_dropout, torch.rand_like: rand_like}


computation_op_map = {
    nn.Conv2d: fuse_conv_unary_eval,
    nn.Linear: fuse_linear_unary_eval,
}


pointwise_op_map = {
    "relu": UnaryFusionOp(nn.ReLU),
    "sigmoid": UnaryFusionOp(nn.Sigmoid),
    "tanh": UnaryFusionOp(nn.Tanh),
    "hardswish": UnaryFusionOp(nn.Hardswish),
    "leaky_relu": UnaryFusionOp(nn.LeakyReLU, scalars=["negative_slope"]),
    "hardtanh": UnaryFusionOp(nn.Hardtanh, scalars=["min_val", "max_val"]),
    "gelu": UnaryFusionOp(nn.GELU, algorithm="approximate"),
}


binary_attr = {
    torch.add: "add",
    torch.Tensor.add: "add",
    operator.add: "add",
    torch.sub: "sub",
    torch.Tensor.sub: "sub",
    operator.sub: "sub",
}


computation_op_binary_op_fusion_map = {
    nn.Conv2d: fuse_conv_binary_eval,
    nn.Linear: fuse_linear_binary_eval,
}


# For add: we support conv/linear + other and other + conv
# For sub, we only support conv/linear - sub
supported_index_list = {
    "add": [
        {"index_computation": 0, "index_pointwise": 1},
        {"index_computation": 1, "index_pointwise": 0},
    ],
    "sub": [{"index_computation": 0, "index_pointwise": 1}],
}

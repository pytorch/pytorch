import copy
import itertools
import logging
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


def fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
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
                conv = modules[node.args[0].target]
                unary = modules[node.target]
                eval_mode = all(not n.training for n in [conv, unary])
                if not eval_mode:
                    continue
                fused_conv = fuse_func(conv, unary, pointwise_name, pointwise_info)
                replace_node_module(node.args[0], modules, fused_conv)
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


computation_op_map = {nn.Conv2d: fuse_conv_unary_eval}


pointwise_op_map = {
    "relu": UnaryFusionOp(nn.ReLU),
    "sigmoid": UnaryFusionOp(nn.Sigmoid),
    "tanh": UnaryFusionOp(nn.Tanh),
    "hardswish": UnaryFusionOp(nn.Hardswish),
    "leaky_relu": UnaryFusionOp(nn.LeakyReLU, scalars=["negative_slope"]),
    "hardtanh": UnaryFusionOp(nn.Hardtanh, scalars=["min_val", "max_val"]),
    "gelu": UnaryFusionOp(nn.GELU, algorithm="approximate"),
}

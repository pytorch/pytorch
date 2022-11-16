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
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.utils.fusion import fuse_conv_bn_eval
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


class UnaryAttr(object):
    def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
        self.op_name = op_name
        self.scalars_attr = scalars_attr if scalars_attr else []
        self.algorithm_attr = algorithm_attr if algorithm_attr else ""
        super(UnaryAttr, self).__init__()

    def __call__(self, unary_module: nn.Module):
        assert all(hasattr(unary_module, item) for item in self.scalars_attr)
        scalars = [getattr(unary_module, item) for item in self.scalars_attr]

        algorithm = ""
        if self.algorithm_attr:
            assert hasattr(unary_module, self.algorithm_attr)
            algorithm = getattr(unary_module, self.algorithm_attr)

        return self.op_name, scalars, algorithm


class ConvUnary2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        unary: nn.Module,
    ):
        super(ConvUnary2d, self).__init__(
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
        self._update_module_params(conv, unary)

    def _update_module_params(self, conv, unary):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.attr, self.scalars, self.algorithm = unary_modules_map[unary.__class__](
            unary
        )

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
        conv: nn.Module,
        binary_op_name: str,
    ):
        super(ConvBinary2d, self).__init__(
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
        self._update_module_params(conv, binary_op_name)

    def _update_module_params(self, conv, binary_op_name):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.binary_attr = binary_op_name
        self.binary_alpha = None
        self.unary_attr = None
        self.unary_scalars = []
        self.unary_algorithm = None

    def _update_unary_params(self, unary):
        self.unary_attr, self.unary_scalars, self.unary_algorithm = unary_modules_map[
            unary.__class__
        ](unary)

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
                self.binary_attr,
                self.binary_alpha,
                self.unary_attr,
                self.unary_scalars,
                self.unary_algorithm,
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
            self.binary_attr,
            self.binary_alpha,
            self.unary_attr,
            self.unary_scalars,
            self.unary_algorithm,
        )

    def forward(self, input, other):
        return self._conv_forward(input, other, self.weight, self.bias)


class ConvBinaryInplace2d(nn.Conv2d):
    def __init__(
        self,
        conv: nn.Module,
        binary_op_name: str,
    ):
        super(ConvBinaryInplace2d, self).__init__(
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
        self._update_module_params(conv, binary_op_name)

    def _update_module_params(self, conv, binary_op_name):
        self.__dict__ = copy.deepcopy(conv.__dict__)
        self.binary_attr = binary_op_name
        self.binary_alpha = None
        self.unary_attr = None
        self.unary_scalars = []
        self.unary_algorithm = None

    def _update_unary_params(self, unary):
        self.unary_attr, self.unary_scalars, self.unary_algorithm = unary_modules_map[
            unary.__class__
        ](unary)

    def _conv_forward(self, input, other, weight, bias):
        if self.padding_mode != "zeros":
            return torch.ops.mkldnn._convolution_pointwise_(
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
                self.binary_attr,
                self.binary_alpha,
                self.unary_attr,
                self.unary_scalars,
                self.unary_algorithm,
            )
        return torch.ops.mkldnn._convolution_pointwise_(
            input,
            other,
            weight,
            bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.binary_attr,
            self.binary_alpha,
            self.unary_attr,
            self.unary_scalars,
            self.unary_algorithm,
        )

    def forward(self, input, other):
        return self._conv_forward(input, other, self.weight, self.bias)


class LinearUnary(nn.Linear):
    def __init__(
        self,
        linear: nn.Module,
        unary: nn.Module,
    ):
        super(LinearUnary, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self._update_module_params(linear, unary)

    def _update_module_params(self, linear, unary):
        self.__dict__ = copy.deepcopy(linear.__dict__)
        self.attr, self.scalars, self.algorithm = unary_modules_map[unary.__class__](
            unary
        )

    def forward(self, input):
        y = torch.ops.mkldnn._linear_pointwise(
            input, self.weight, self.bias, self.attr, self.scalars, self.algorithm
        )
        return y


class LinearBinary(nn.Linear):
    def __init__(self, linear: nn.Module, binary_op_name: str):
        super(LinearBinary, self).__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
            linear.weight.dtype,
        )
        self._update_module_params(linear, binary_op_name)

    def _update_module_params(self, linear, binary_op_name):
        self.__dict__ = copy.deepcopy(linear.__dict__)

        self.attr = binary_op_name

    def forward(self, input, other):
        y = torch.ops.mkldnn._linear_pointwise(
            input, other, self.weight, self.bias, self.attr
        )
        return y


def fused_conv_unary_eval(conv: nn.Module, unary: nn.Module):
    assert not (conv.training), "Fusion only for eval!"
    return ConvUnary2d(
        conv,
        unary,
    )


def fused_conv_binary_eval(conv: nn.Module, binary_op_name: str):
    assert not (conv.training), "Fusion only for eval!"
    return ConvBinary2d(
        conv,
        binary_op_name,
    )


def fused_conv_binary_inplace_eval(conv: nn.Module, binary_op_name: str):
    assert not (conv.training), "Fusion only for eval!"
    return ConvBinaryInplace2d(
        conv,
        binary_op_name,
    )


def fused_binary_unary_eval(conv_binary: nn.Module, unary: nn.Module):
    assert not (conv_binary.training), "Fusion only for eval!"
    # reuse origin conv module, and just update its' unary attr.
    conv_binary._update_unary_params(unary)
    return conv_binary


def is_bfloat16_module(m):
    weight_is_bf16 = m.weight.dtype == torch.bfloat16
    bias_is_bf16 = m.bias is None or m.bias.dtype == torch.bfloat16
    return weight_is_bf16 and bias_is_bf16


def fused_linear_unary_eval(linear: nn.Module, unary: nn.Module):
    assert not (linear.training), "Fusion only for eval!"
    return LinearUnary(
        linear,
        unary,
    )


def fused_linear_binary_eval(linear: nn.Module, attr: str):
    assert not (linear.training), "Fusion only for eval!"
    linear_binary = LinearBinary(
        linear,
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
    return (
        (node.op == "call_function" and node.target in [torch.add, torch.sub])
        or (
            node.op == "call_function"
            and node.target
            in [operator.add, operator.iadd, operator.sub, operator.isub]
        )
        or (node.op == "call_method" and node.target in ["add", "add_", "sub", "sub_"])
    )


def check_binary_op_kwargs_is_default(node):
    # For binary op, we hope the kwargs values are the default value:
    # torch.sub(add)(input, other, *, alpha=1, out=None).
    if len(node.args) > 2:
        return False
    if len(node.kwargs) > 0:
        if "out" in node.kwargs and node.kwargs["out"] is not None:
            return False
        if "alpha" in node.kwargs and node.kwargs["alpha"] != 1.0:
            return False
    return True


def check_node_is_add_inplace(node):
    return (node.op == "call_function" and node.target in [operator.iadd]) or (
        node.op == "call_method" and node.target in ["add_"]
    )


def fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    # make sure the autograd is disabled.
    if torch.is_grad_enabled():
        return gm
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
    is_cpu = all(
        example_input.device == torch.device("cpu") for example_input in example_inputs
    )
    if not is_cpu:
        return gm
    gm = fuse_conv_bn(gm)
    # For binary fusion, we need to check inputs info to make sure
    # the binary inputs have same tensor info(device, dtype, and layout).
    ShapeProp(gm).propagate(*example_inputs)
    gm = fuse_unary(gm)
    gm = fuse_binary_inplace(gm)
    gm = fuse_binary(gm)
    # why re-run fuse_unary? we want to enable conv+binary+unary fusion,
    # such as conv+add+relu for vision model.
    gm = fuse_unary(gm)

    return gm


def fuse_conv_bn(gm: torch.fx.GraphModule, inplace=False):
    """
    Fuses Convolution/BN layers for inference purposes.
    """
    patterns = [
        (torch.nn.Conv1d, torch.nn.BatchNorm1d),
        (torch.nn.Conv2d, torch.nn.BatchNorm2d),
        (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    ]
    modules = dict(gm.named_modules())

    for pattern in patterns:
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                eval_mode = all(not n.training for n in [conv, bn])
                if not eval_mode:
                    continue
                if not bn.track_running_stats:
                    continue
                fused_conv = fuse_conv_bn_eval(conv, bn)
                replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                gm.graph.erase_node(node)
                gm.graph.lint()
    gm.recompile()
    return gm


def fuse_unary(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())

    for (unary_module, _), (computation_module, fuse_func,) in itertools.product(
        unary_modules_map.items(), computation_op_unary_op_fusion_map.items()
    ):
        pattern = (computation_module, unary_module)
        for node in gm.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if (
                    len(node.args[0].users) > 1
                ):  # Output of computation_node is used by other nodes
                    continue
                computation_node = modules[node.args[0].target]
                unary_node = modules[node.target]
                eval_mode = all(not n.training for n in [computation_node, unary_node])
                if not eval_mode:
                    continue
                # TODO: support padding str input("valid", "same").
                if type(computation_node) in [nn.Conv2d] and isinstance(
                    computation_node.padding, str
                ):
                    continue
                # only fuse for linear when the dtype is bf16
                if type(computation_node) in [nn.Linear] and not is_bfloat16_module(
                    computation_node
                ):
                    continue
                fused_module = fuse_func(computation_node, unary_node)
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


def binary_inputs_meta_is_same(binary_node):
    tensor0_meta = binary_node.args[0].meta.get("tensor_meta")
    tensor1_meta = binary_node.args[1].meta.get("tensor_meta")
    if not tensor0_meta or not tensor1_meta:
        return False
    if (
        tensor0_meta.shape != tensor1_meta.shape
        or tensor0_meta.stride != tensor1_meta.stride
        or tensor0_meta.dtype != tensor1_meta.dtype
    ):
        return False

    return True


def fuse_binary(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if check_node_is_binary(node) and check_binary_op_kwargs_is_default(node):
            for node_kind, fuse_func in computation_op_binary_op_fusion_map.items():
                if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
                    node.args[1], torch.fx.Node
                ):
                    continue
                if not binary_inputs_meta_is_same(node):
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
                        # TODO: support padding str input("valid", "same").
                        if type(computation_node) in [nn.Conv2d] and isinstance(
                            computation_node.padding, str
                        ):
                            continue
                        # only fuse for linear when the dtype is bf16
                        if type(computation_node) in [
                            nn.Linear
                        ] and not is_bfloat16_module(computation_node):
                            continue
                        replace_and_fuse_for_binary(
                            computation_node,
                            node,
                            fuse_func,
                            attr if attr != "iadd" else "add",
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


def fuse_binary_inplace(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if check_node_is_add_inplace(node) and check_binary_op_kwargs_is_default(node):
            for (
                node_kind,
                fuse_func,
            ) in computation_op_binary_op_fusion_inplace_map.items():
                if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
                    node.args[1], torch.fx.Node
                ):
                    continue
                if not binary_inputs_meta_is_same(node):
                    continue
                if check_node_kind(node.args[1], modules, node_kind):
                    if len(node.args[1].users) > 1:
                        continue
                    # make sure the output and input are not same tensor.
                    if node.args[1].args[0] == node.args[0]:
                        continue
                    computation_node = modules[node.args[1].target]
                    # TODO: support padding str input("valid", "same").
                    if type(computation_node) in [nn.Conv2d] and isinstance(
                        computation_node.padding, str
                    ):
                        continue
                    replace_and_fuse_for_binary(
                        computation_node,
                        node,
                        fuse_func,
                        "add",
                        modules,
                        1,  # conv module index
                        0,  # binary op index
                    )
                    # Make sure the fused node is post node of node's inputs nodes.
                    node.append(node.args[1])
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
def lowmem_dropout(input, p=0.5, training=True, inplace=False):
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


computation_op_unary_op_fusion_map = {
    nn.Conv2d: fused_conv_unary_eval,
    nn.Linear: fused_linear_unary_eval,
    ConvBinary2d: fused_binary_unary_eval,
    ConvBinaryInplace2d: fused_binary_unary_eval,
}


unary_modules_map = {
    nn.ReLU: UnaryAttr("relu"),
    nn.Sigmoid: UnaryAttr("sigmoid"),
    nn.Tanh: UnaryAttr("tanh"),
    nn.Hardswish: UnaryAttr("hardswish"),
    nn.LeakyReLU: UnaryAttr("leaky_relu", scalars_attr=["negative_slope"]),
    nn.Hardtanh: UnaryAttr("hardtanh", scalars_attr=["min_val", "max_val"]),
    nn.GELU: UnaryAttr("gelu", algorithm_attr="approximate"),
}


binary_attr = {
    torch.add: "add",  # node.op == "call_function"
    "add": "add",  # node.op == "call_method"
    "add_": "iadd",  # node.op == "call_method"
    operator.add: "add",  # node.op == "call_function"
    operator.iadd: "iadd",  # node.op == "call_function"
    torch.sub: "sub",  # node.op == "call_function"
    "sub": "sub",  # node.op == "call_method"
    "sub_": "sub",  # node.op == "call_method"
    operator.sub: "sub",  # node.op == "call_function"
    operator.isub: "sub",  # node.op == "call_function"
}


computation_op_binary_op_fusion_map = {
    nn.Conv2d: fused_conv_binary_eval,
    nn.Linear: fused_linear_binary_eval,
}


computation_op_binary_op_fusion_inplace_map = {
    nn.Conv2d: fused_conv_binary_inplace_eval,
}

# For add: we support conv/linear + other and other + conv
# For sub/add_/sub_, we only support conv/linear - other
# or conv/linear +(-)= other
supported_index_list = {
    "add": [
        {"index_computation": 0, "index_pointwise": 1},
        {"index_computation": 1, "index_pointwise": 0},
    ],
    "iadd": [{"index_computation": 0, "index_pointwise": 1}],
    "sub": [{"index_computation": 0, "index_pointwise": 1}],
}

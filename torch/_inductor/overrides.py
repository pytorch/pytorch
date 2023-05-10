import logging
import random
import weakref

import functorch

import torch
from torch import _prims
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode
from .utils import is_cpu_device
from . import config
from .utils import decode_device, is_cpu_device

log = logging.getLogger(__name__)


class AutogradMonkeypatch(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        return replace_fn(func, is_cpu_device(args))(*args, **kwargs)


patch_functions = AutogradMonkeypatch


def replace_fx(gm: torch.fx.GraphModule, example_inputs):
    # Sometimes patch_functions() misses things already in the graph
    changed = 0
    is_cpu = is_cpu_device(example_inputs)

    for node in reversed(list(gm.graph.nodes)):
        if (
            node.op == "call_function"
            and replace_fn(node.target, is_cpu) is not node.target
        ):
            with gm.graph.inserting_before(node):
                node.replace_all_uses_with(
                    gm.graph.call_function(
                        replace_fn(node.target, is_cpu), node.args, node.kwargs
                    )
                )
            gm.graph.erase_node(node)
            changed += 1

    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm


def is_quantized_graph_module(gm: torch.fx.GraphModule):
    found_quantize = False
    quantize_ops = (
        torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
    )
    for node in gm.graph.nodes:
        if node.target in quantize_ops:
            found_quantize = True
            break
    return found_quantize


def _quantize_and_replace_weight(
    gm: torch.fx.GraphModule, dq_per_channel_node: torch.fx.Node
):
    # pattern: w - q - dq - weighted op
    q_per_channel_node = dq_per_channel_node.args[0]
    weight_node = q_per_channel_node.args[0]
    w_attr_name = weight_node.target
    weight = getattr(gm, w_attr_name)

    assert isinstance(weight, torch.Tensor), "Cannot find weight for quantization"
    if weight.is_quantized:
        return
    quantize_args = (
        getattr(gm, n.target) if isinstance(n, torch.fx.Node) else n
        for n in q_per_channel_node.args
    )
    q_arg_list = list(quantize_args)
    q_arg_tuple = tuple(q_arg_list)
    weight_int8 = torch.nn.parameter.Parameter(
        torch.ops.quantized_decomposed.quantize_per_channel(*q_arg_tuple),
        requires_grad=False,
    )

    qw_attr_name = w_attr_name + "_quant"
    setattr(gm, qw_attr_name, weight_int8)
    weight_node.target = qw_attr_name
    gm.graph.owning_module._buffers[qw_attr_name] = weight_int8
    delattr(gm, w_attr_name)
    q_per_channel_node.replace_all_uses_with(weight_node)
    gm.graph.erase_node(q_per_channel_node)


def pre_quantize_weights(gm: torch.fx.GraphModule):
    # pattern: w - q - dq - weighted op
    aten = torch.ops.aten
    decomposed = torch.ops.quantized_decomposed
    for node in gm.graph.nodes:
        dq_per_channel_node = None
        if node.target == aten.convolution.default:
            # conv args = (x, w, ...)
            dq_per_channel_node = node.args[1]
        if dq_per_channel_node is not None:
            assert (
                dq_per_channel_node.target == decomposed.dequantize_per_channel
            ), "Cannot find the dequantize op for weight"
            _quantize_and_replace_weight(gm, dq_per_channel_node)
    gm.graph.lint()
    gm.recompile()


def fuse_quantization(gm: torch.fx.GraphModule, example_inputs):
    # skip if gm is not a quantized graph module
    if not (is_quantized_graph_module(gm) and is_cpu_device(example_inputs)):
        return gm
    gm.graph.eliminate_dead_code()
    gm.recompile()
    # Fuse `quant_per_channel - weight` and replace the original fp32 weight with quantized one
    pre_quantize_weights(gm)

    return gm


def _philox_rand_like_meta(input, seed, offset):
    return _prims.TensorMeta(input)


def _philox_rand_like(input, seed, offset):
    # placeholder only used in tracing
    return torch.rand_like(input)


philox_rand_like = _prims._make_prim(
    schema="philox_rand_like(Tensor input, Tensor seed, SymInt offset) -> Tensor",
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
    def get_seed_offset(cls, x, device=None):
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

        if device is None:
            device = x.device
        device = decode_device(device)
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
        scale = float(0.0) if p == 1.0 else float(1.0 / (1.0 - p))
        seed, offset = PhiloxRandomState.get_seed_offset(x)
        ctx.save_for_backward(seed)
        ctx.offset = offset
        bool_mask = philox_rand_like(x, seed, offset) > p
        return bool_mask.to(x.dtype) * x * scale

    @staticmethod
    def backward(ctx, grad_output):
        p = ctx.p
        scale = float(0.0) if p == 1.0 else float(1.0 / (1.0 - p))
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
    device = kwargs.get("device", x.device)
    seed, offset = PhiloxRandomState.get_seed_offset(x, device)
    return philox_rand_like(x.to(device), seed, offset).to(
        kwargs.get("dtype", torch.float32)
    )


def replace_fn(fn, is_cpu):
    """
    Perform any applicable replacements on `fn`
    """
    if config.fallback_random:
        return fn
    if config.lowmem_dropout and fn is torch.nn.functional.dropout and not is_cpu:
        return lowmem_dropout

    replacements = {}
    # TODO: Revisit the functionalize_rng_ops for lowmem dropout
    if not functorch.compile.config.functionalize_rng_ops:
        replacements.update({torch.rand_like: rand_like})
    return replacements.get(fn, fn)

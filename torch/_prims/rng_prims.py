# mypy: allow-untyped-defs
from typing import cast

import torch
import torch.utils._pytree as pytree
from torch import _prims
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.types import _device, _dtype


def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


def register_rng_prim(name, schema, impl_aten, impl_meta, doc, tags=None):
    rngprim_def = torch.library.custom_op(
        "rngprims::" + name, impl_aten, mutates_args=(), schema=schema
    )

    rngprim_def.register_fake(impl_meta)

    prim_packet = getattr(torch._ops.ops.rngprims, name)
    prim = prim_packet.default
    if tags:
        prim._tags = tags

    for p in (prim_packet, prim):
        p.__doc__ = doc
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]

        p.schema = name + schema
        p.impl_aten = impl_aten
        p.prim_meta_impl = impl_meta


# Philox rand offsets could be shared in future with other philox ops, so
# keeping these functions in global scope.
def philox_rand_offset_meta(
    shape: torch.Size,
):
    return _prims.TensorLike(torch.tensor(0, dtype=torch.int64))


def philox_rand_offset(
    shape: torch.Size,
):
    # For impl, look at the function calc_execution_policy in the file
    # aten/src/ATen/native/cuda/DistributionTemplates.h. The impl was copied at
    # commit hash 72aa0667bd16707d50eb8fa337092a1f5d11dfb6
    numel_scalar = 1
    for dim_size in shape:
        numel_scalar *= dim_size
    numel = torch.scalar_tensor(numel_scalar, dtype=torch.int64)

    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
    blocks_per_sm = device_property.max_threads_per_multi_processor // block_size
    num = cast(int, numel)
    grid_size = (num + block_size - 1) // block_size
    grid_size = min(grid_size, device_property.multi_processor_count * blocks_per_sm)
    return ((num - 1) // (block_size * grid_size * unroll) + 1) * curand4_engine_calls


def register_philox_rand():
    name = "philox_rand"
    schema = "(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)"  # noqa: B950

    def _philox_rand_meta(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: tuple[int, ...] | None,
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, its unused.
        if stride is not None:
            raise AssertionError(f"stride must be None, got {stride}")
        stride = make_contiguous_strides_for(shape)
        random_values = _prims.TensorMeta(
            shape=shape, strides=stride, dtype=dtype, device=device
        )
        offset = philox_rand_offset_meta(shape)
        return (random_values, offset)

    def _philox_rand(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: tuple[int, ...] | None,
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, its unused.
        if stride is not None:
            raise AssertionError(f"stride must be None, got {stride}")
        if device.type == "cpu":
            devices = []
        else:
            devices = [device]

        if device.type != "cuda":
            raise throw_on_non_cuda(device)

        with torch.random.fork_rng(devices):
            CUDARngStateHelper.set_torch_state_tensor(seed, offset)
            random_values = torch.rand(shape, device=device, dtype=dtype)

        return random_values, philox_rand_offset(shape)

    register_rng_prim(
        name=name,
        schema=schema,
        impl_aten=_philox_rand,
        impl_meta=_philox_rand_meta,
        doc="Philox based stateless rand operator",
        tags=(torch.Tag.nondeterministic_seeded,),
    )


def get_device(args, kwargs):
    if kwargs.get("device"):
        device = kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
        return device.type

    devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
    if any(dev == "cuda" for dev in devices):
        return "cuda"
    elif any(dev == "xpu" for dev in devices):
        return "xpu"
    elif any(dev == "hpu" for dev in devices):
        return "hpu"
    elif any(dev == "cpu" for dev in devices):
        return "cpu"
    return None


def register_run_and_save_rng_state_op():
    class RunAndSaveRngState(HigherOrderOperator):
        def __init__(self):
            super().__init__("run_and_save_rng_state", cacheable=True)

        def __call__(self, op, *args, **kwargs):
            # pyrefly: ignore [missing-attribute]
            return super().__call__(op, *args, **kwargs)

    run_and_save_rng_state = RunAndSaveRngState()

    run_and_save_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_and_save_rng_state, deferred_error=True)
    )

    @run_and_save_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(op, *args, **kwargs):
        return torch.cuda.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(op, *args, **kwargs):
        return torch.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.HPU)
    def impl_hpu(op, *args, **kwargs):
        if hasattr(torch, "hpu"):
            return torch.hpu.get_rng_state(), op(*args, **kwargs)
        raise RuntimeError("functionalize a hpu RNG operator is not supported.")

    @run_and_save_rng_state.py_impl(DispatchKey.XPU)
    def impl_xpu(op, *args, **kwargs):
        return torch.xpu.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, **kwargs):
        impl_map = {
            "cuda": impl_cuda,
            "cpu": impl_cpu,
            "hpu": impl_hpu,
            "xpu": impl_xpu,
        }
        device = get_device(args, kwargs)
        if device not in impl_map:
            raise AssertionError(f"Backend not supported for {device}")
        impl = impl_map[device]
        return impl(op, *args, **kwargs)

    @run_and_save_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, op, *args, **kwargs):
        # Check device to call the right impl
        with mode:
            return impl_backend_select(op, *args, **kwargs)

    @run_and_save_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, op, *args, **kwargs):
        out = impl_backend_select(op, *args, **kwargs)
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
        out_proxy = mode.tracer.create_proxy(
            "call_function", run_and_save_rng_state, proxy_args, proxy_kwargs
        )
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

    return run_and_save_rng_state


def register_run_with_rng_state_op():
    class RunWithRngState(HigherOrderOperator):
        def __init__(self):
            super().__init__("run_with_rng_state", cacheable=True)

        def __call__(self, rng_state, op, *args, **kwargs):
            # pyrefly: ignore [missing-attribute]
            return super().__call__(rng_state, op, *args, **kwargs)

    run_with_rng_state = RunWithRngState()

    run_with_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_with_rng_state, deferred_error=True)
    )

    @run_with_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(rng_state, op, *args, **kwargs):
        current_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state.cpu())
        out = op(*args, **kwargs)
        torch.cuda.set_rng_state(current_state)
        return out

    @run_with_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(rng_state, op, *args, **kwargs):
        current_state = torch.get_rng_state()
        torch.set_rng_state(rng_state)
        out = op(*args, **kwargs)
        torch.set_rng_state(current_state)
        return out

    @run_with_rng_state.py_impl(DispatchKey.HPU)
    def impl_hpu(rng_state, op, *args, **kwargs):
        if hasattr(torch, "hpu"):
            current_state = torch.hpu.get_rng_state()
            torch.hpu.set_rng_state(rng_state)
            out = op(*args, **kwargs)
            torch.hpu.set_rng_state(current_state)
            return out
        raise RuntimeError("functionalize a hpu RNG operator is not supported.")

    @run_with_rng_state.py_impl(DispatchKey.XPU)
    def impl_xpu(rng_state, op, *args, **kwargs):
        current_state = torch.xpu.get_rng_state()
        torch.xpu.set_rng_state(rng_state)
        out = op(*args, **kwargs)
        torch.xpu.set_rng_state(current_state)
        return out

    @run_with_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, rng_state, op, *args, **kwargs):
        # TODO: you don't need to do this, the dispatch here already disabled
        # it
        with disable_proxy_modes_tracing():
            out = run_with_rng_state(rng_state, op, *args, **kwargs)
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (rng_state, op, *args))
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
        out_proxy = mode.tracer.create_proxy(
            "call_function", run_with_rng_state, proxy_args, proxy_kwargs
        )
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

    @run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(rng_state, op, *args, **kwargs):
        impl_map = {
            "cuda": impl_cuda,
            "cpu": impl_cpu,
            "hpu": impl_hpu,
            "xpu": impl_xpu,
        }
        device = get_device(args, kwargs)
        if device not in impl_map:
            raise AssertionError(f"Backend not supported for {device}")
        impl = impl_map[device]
        return impl(rng_state, op, *args, **kwargs)

    @run_with_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, rng_state, op, *args, **kwargs):
        # Skip setting the set_rng_state as it does not work well with fake tensors.
        # And it does not matter for the fake tensor mode.
        with mode:
            return op(*args, **kwargs)

    @run_with_rng_state.py_functionalize_impl
    def impl_functional(ctx, rng_state, op, *args, **kwargs):
        unwrapped_rng_state = ctx.unwrap_tensors(rng_state)
        unwrapped_args = ctx.unwrap_tensors(args)
        unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

        with ctx.redispatch_to_next():
            out = run_with_rng_state(
                unwrapped_rng_state, op, *unwrapped_args, **unwrapped_kwargs
            )
            return ctx.wrap_tensors(out)

    return run_with_rng_state


run_and_save_rng_state = register_run_and_save_rng_state_op()
run_with_rng_state = register_run_with_rng_state_op()


def register_graphsafe_run_with_rng_state_op():
    class GraphSafeRunWithRngState(HigherOrderOperator):
        def __init__(self):
            super().__init__("graphsafe_run_with_rng_state")

        def __call__(self, op, *args, rng_state=None, **kwargs):
            # pyrefly: ignore [missing-attribute]
            return super().__call__(op, *args, rng_state=rng_state, **kwargs)

    graphsafe_run_with_rng_state = GraphSafeRunWithRngState()

    graphsafe_run_with_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(graphsafe_run_with_rng_state, deferred_error=True)
    )

    @graphsafe_run_with_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(op, *args, rng_state=None, **kwargs):
        # pyrefly: ignore [missing-attribute]
        device_idx = rng_state.device.index
        generator = torch.cuda.default_generators[device_idx]
        current_state = generator.graphsafe_get_state()

        generator.graphsafe_set_state(rng_state)
        out = op(*args, **kwargs)
        generator.graphsafe_set_state(current_state)
        return out

    @graphsafe_run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, rng_state=None, **kwargs):
        device = get_device(args, kwargs)
        if device != "cuda":
            raise AssertionError(
                f"GraphSafe RNG operations only supported for CUDA, got {device}"
            )
        return impl_cuda(op, *args, rng_state=rng_state, **kwargs)

    @graphsafe_run_with_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, op, *args, rng_state=None, **kwargs):
        with mode:
            return op(*args, **kwargs)

    @graphsafe_run_with_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, op, *args, rng_state=None, **kwargs):
        with disable_proxy_modes_tracing():
            out = graphsafe_run_with_rng_state(op, *args, rng_state=rng_state, **kwargs)
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
        proxy_kwargs = pytree.tree_map(
            mode.tracer.unwrap_proxy, {"rng_state": rng_state, **kwargs}
        )
        out_proxy = mode.tracer.create_proxy(
            "call_function", graphsafe_run_with_rng_state, proxy_args, proxy_kwargs
        )
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

    @graphsafe_run_with_rng_state.py_functionalize_impl
    def impl_functional(ctx, op, *args, rng_state=None, **kwargs):
        unwrapped_rng_state = (
            ctx.unwrap_tensors(rng_state) if rng_state is not None else None
        )
        unwrapped_args = ctx.unwrap_tensors(args)
        unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

        with ctx.redispatch_to_next():
            out = graphsafe_run_with_rng_state(
                op, *unwrapped_args, rng_state=unwrapped_rng_state, **unwrapped_kwargs
            )
            return ctx.wrap_tensors(out)

    return graphsafe_run_with_rng_state


graphsafe_run_with_rng_state = register_graphsafe_run_with_rng_state_op()


def register_rng_prims():
    register_philox_rand()

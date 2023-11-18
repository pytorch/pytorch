from typing import Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import _prims
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator

from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.types import _device, _dtype


rngprim_namespace = "rngprims"
rngprim = torch.library.Library(rngprim_namespace, "DEF")
rngprim_impl = torch.library.Library(
    rngprim_namespace, "IMPL", "CompositeExplicitAutograd"
)
rngprim_autograd_impl = torch.library.Library(rngprim_namespace, "IMPL", "Autograd")
rngprim_meta_impl = torch.library.Library(rngprim_namespace, "IMPL", "Meta")


def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


def register_rng_prim(name, schema, impl_aten, impl_meta, doc, tags=None):
    rngprim.define(schema)
    rngprim_impl.impl(name, impl_aten)
    rngprim_meta_impl.impl(name, impl_meta)

    prim_packet = getattr(torch._ops.ops.rngprims, name)
    prim = prim_packet.default
    if tags:
        prim._tags = tags

    rngprim_autograd_impl.impl(name, backwards_not_supported(prim))

    for p in (prim_packet, prim):
        p.__doc__ = doc
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]

        p.schema = schema
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
    grid_size = (numel + block_size - 1) // block_size
    grid_size = min(grid_size, device_property.multi_processor_count * blocks_per_sm)
    offset = (
        (numel - 1) // (block_size * grid_size * unroll) + 1
    ) * curand4_engine_calls
    return offset


def register_philox_rand():
    name = "philox_rand"
    schema = "philox_rand(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)"  # noqa: B950

    def _philox_rand_meta(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: Optional[Tuple[int, ...]],
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, its unused.
        assert stride is None
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
        stride: Optional[Tuple[int, ...]],
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, its unused.
        assert stride is None
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
    elif any(dev == "cpu" for dev in devices):
        return "cpu"
    return None


def register_run_and_save_rng_state_op():
    run_and_save_rng_state = HigherOrderOperator("run_and_save_rng_state")

    run_and_save_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_and_save_rng_state, deferred_error=True)
    )

    @run_and_save_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(op, *args, **kwargs):
        return torch.cuda.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(op, *args, **kwargs):
        return torch.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, **kwargs):
        impl_map = {"cuda": impl_cuda, "cpu": impl_cpu}
        device = get_device(args, kwargs)
        assert device in impl_map, f"Backend not supported for {device}"
        impl = impl_map[device]
        return impl(op, *args, **kwargs)

    @run_and_save_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, op, *args, **kwargs):
        # Check device to call the right impl
        with mode:
            return impl_backend_select(op, *args, **kwargs)

    @run_and_save_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, op, *args, **kwargs):
        if mode.enable_tracing:
            out = impl_backend_select(op, *args, **kwargs)
            proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
            proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
            out_proxy = mode.tracer.create_proxy(
                "call_function", run_and_save_rng_state, proxy_args, proxy_kwargs
            )
            return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
        else:
            return run_and_save_rng_state(op, *args, **kwargs)

    return run_and_save_rng_state


def register_run_with_rng_state_op():
    run_with_rng_state = HigherOrderOperator("run_with_rng_state")

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

    @run_with_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, rng_state, op, *args, **kwargs):
        if mode.enable_tracing:
            with disable_proxy_modes_tracing():
                out = run_with_rng_state(rng_state, op, *args, **kwargs)
            proxy_args = pytree.tree_map(
                mode.tracer.unwrap_proxy, (rng_state, op, *args)
            )
            proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
            out_proxy = mode.tracer.create_proxy(
                "call_function", run_with_rng_state, proxy_args, proxy_kwargs
            )
            return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
        else:
            return run_with_rng_state(rng_state, op, *args, **kwargs)

    @run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(rng_state, op, *args, **kwargs):
        impl_map = {"cuda": impl_cuda, "cpu": impl_cpu}
        device = get_device(args, kwargs)
        assert device in impl_map, f"Backend not supported for {device}"
        impl = impl_map[device]
        return impl(rng_state, op, *args, **kwargs)

    @run_with_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, rng_state, op, *args, **kwargs):
        # Skip setting the set_rng_state as it does not work well with fake tensors.
        # And it does not matter for the fake tensor mode.
        with mode:
            return op(*args, **kwargs)

    return run_with_rng_state


run_and_save_rng_state = register_run_and_save_rng_state_op()
run_with_rng_state = register_run_with_rng_state_op()


def register_rng_prims():
    register_philox_rand()

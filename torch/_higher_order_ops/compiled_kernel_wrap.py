"""HOPs for embedding a non-Triton compiled fused kernel into an FX graph.

The Inductor FX backend (``config.fx_wrapper``) embeds Triton kernel launches as
``triton_kernel_wrapper_mutation`` nodes. Non-Triton fused kernels -- e.g. an
Inductor CPU ``cpp_fused_*`` pybinding -- are plain callables that take a flat
positional tensor list and write their outputs in place, so they cannot be
launched through the Triton HOP. ``compiled_kernel_wrapper_mutation`` is the
analog for them: the callable is stored in a global side table (callables are not
graphable) and the HOP carries an integer index plus ``mutated_arg_indices``, the
positions in ``args`` the kernel writes. Inductor knows the kernel's output
buffers, so unlike Triton (which parses the kernel's TTIR) the mutated set is
supplied by the producer and functionalization needs no source analysis.

Dispatch parity with ``triton_kernel_wrapper_mutation``: dense
(CompositeExplicitAutograd), FakeTensorMode, Meta, ProxyTorchDispatchMode, and
functionalize, plus a functional sibling ``compiled_kernel_wrapper_functional``.
"""

import threading
from collections.abc import Callable

import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import redirect_to_mode, register_fake
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


class CompiledKernelSideTable:
    """Global idx <-> compiled-callable table, mirroring KernelSideTable."""

    def __init__(self) -> None:
        self.id_to_kernel: dict[int, Callable] = {}
        self.kernel_to_id: dict[Callable, int] = {}
        self.lock = threading.Lock()

    def add_kernel(self, kernel: Callable) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]
            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    def get_kernel(self, idx: int) -> Callable:
        if idx not in self.id_to_kernel:
            raise AssertionError(f"Compiled kernel index {idx} not found")
        return self.id_to_kernel[idx]

    def reset_table(self) -> None:
        self.id_to_kernel = {}
        self.kernel_to_id = {}


compiled_kernel_side_table = CompiledKernelSideTable()


class CompiledKernelWrapperMutation(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("compiled_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
    ) -> None:
        return super().__call__(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=args,
        )


class CompiledKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("compiled_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
    ) -> dict[int, Tensor]:
        return super().__call__(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=args,
        )


compiled_kernel_wrapper_mutation = CompiledKernelWrapperMutation()
compiled_kernel_wrapper_functional = CompiledKernelWrapperFunctional()


def _trace(proxy_mode, func_overload, node_args):
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


# The mutation HOP returns None; its only effect is the in-place write into args.
# Like triton_kernel_wrapper_mutation it is not registered as side-effectful, so a
# bare eliminate_dead_code() would drop it -- but the Inductor FX backend's output
# graph (where these nodes live) is never DCE'd: the eliminate_dead_code passes run
# on the pre-codegen aten graph, not on the converted GraphModule.
@compiled_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def _mutation_dense(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    compiled_kernel_side_table.get_kernel(kernel_idx)(*args)
    return None


@register_fake(compiled_kernel_wrapper_mutation, skip_cache=True)
def _mutation_fake(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    return None


@compiled_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _mutation_meta(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    return None


@compiled_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def _mutation_proxy(
    mode, *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    _trace(
        mode,
        compiled_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "mutated_arg_indices": mutated_arg_indices,
            "args": args,
        },
    )
    return None


@compiled_kernel_wrapper_mutation.py_functionalize_impl
def _mutation_functionalize(
    ctx, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> None:
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        new_vals = compiled_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=unwrapped_args,
        )
    for i, output_arg in new_vals.items():
        input_arg = args[i]
        ctx.replace(input_arg, output_arg)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)
    return None


def _run_functional(kernel_idx, mutated_arg_indices, args):
    new_args = list(args)
    clones: dict[int, Tensor] = {}
    for i in mutated_arg_indices:
        clones[i] = clone_preserve_strides(args[i])
        new_args[i] = clones[i]
    compiled_kernel_side_table.get_kernel(kernel_idx)(*new_args)
    return clones


@compiled_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def _functional_dense(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    return _run_functional(kernel_idx, mutated_arg_indices, args)


@register_fake(compiled_kernel_wrapper_functional, skip_cache=True)
def _functional_fake(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    return {i: clone_preserve_strides(args[i]) for i in mutated_arg_indices}


@compiled_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def _functional_proxy(
    mode, *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    return _trace(
        mode,
        compiled_kernel_wrapper_functional,
        {
            "kernel_idx": kernel_idx,
            "mutated_arg_indices": mutated_arg_indices,
            "args": args,
        },
    )


@compiled_kernel_wrapper_functional.py_functionalize_impl
def _functional_functionalize(
    ctx, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple
) -> dict[int, Tensor]:
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        outputs = compiled_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            mutated_arg_indices=mutated_arg_indices,
            args=unwrapped_args,
        )
    return ctx.wrap_tensors(outputs)


# Fall through the keys these HOPs do not implement so the dispatcher reaches the
# dense / functionalize impls, mirroring triton_kernel_wrapper_mutation.
for _hop in (compiled_kernel_wrapper_mutation, compiled_kernel_wrapper_functional):
    _hop.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.ADInplaceOrView)
    _hop.fallthrough(DispatchKey.BackendSelect)
    _hop.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
    _hop.fallthrough(DispatchKey.AutogradCUDA)
    _hop.fallthrough(DispatchKey.AutogradCPU)


# Redirect the mutation HOP through activation-checkpoint caching modes so it is
# handled under torch.utils.checkpoint, mirroring triton_kernel_wrapper_mutation.
redirect_to_mode(compiled_kernel_wrapper_mutation, _CachingTorchDispatchMode)
redirect_to_mode(compiled_kernel_wrapper_mutation, _CachedTorchDispatchMode)

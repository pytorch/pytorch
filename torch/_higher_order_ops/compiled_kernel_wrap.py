r"""HOPs for launching non-Triton compiled kernels from Inductor FXIR.

The side table holds compiled callables, such as CPU ``cpp_fused_*`` bindings.
FXIR supplies their positional arguments and the indices of the arguments
written by the kernel.
"""

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.user_kernel_hop import (
    clone_mutated_arg,
    KernelSideTable,
    register_user_kernel_hop_dispatch,
    trace_user_kernel_wrapper,
    UserKernelWrapperHOP,
)
from torch._higher_order_ops.utils import register_fake
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


compiled_kernel_side_table: KernelSideTable[Callable[..., Any]] = KernelSideTable()


class CompiledKernelWrapperMutation(UserKernelWrapperHOP):
    def __init__(self) -> None:
        # The callable lives in the current process's side table.
        super().__init__("compiled_kernel_wrapper_mutation", cacheable=False)

    def __call__(
        self,
        kernel_idx: int,
        mutated_arg_indices: tuple[int, ...],
        args: tuple[Any, ...],
    ) -> None:
        return super().__call__(
            kernel_idx=kernel_idx, mutated_arg_indices=mutated_arg_indices, args=args
        )


class CompiledKernelWrapperFunctional(UserKernelWrapperHOP):
    def __init__(self) -> None:
        super().__init__("compiled_kernel_wrapper_functional", cacheable=False)

    def __call__(
        self,
        kernel_idx: int,
        mutated_arg_indices: tuple[int, ...],
        args: tuple[Any, ...],
        *,
        tensor_bases: dict[int, Tensor] | None = None,
    ) -> dict[int, Tensor]:
        hop_kwargs: dict[str, Any] = {
            "kernel_idx": kernel_idx,
            "mutated_arg_indices": mutated_arg_indices,
            "args": args,
        }
        if tensor_bases:
            hop_kwargs["tensor_bases"] = tensor_bases
        return super().__call__(**hop_kwargs)


compiled_kernel_wrapper_mutation = CompiledKernelWrapperMutation()
compiled_kernel_wrapper_functional = CompiledKernelWrapperFunctional()


# FXIR emits this HOP after Inductor's dead-code elimination passes.
@compiled_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def _mutation_dense(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple[Any, ...]
) -> None:
    compiled_kernel_side_table.get_kernel(kernel_idx)(*args)


@compiled_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
@register_fake(compiled_kernel_wrapper_mutation, skip_cache=True)
def _mutation_fake(
    *, kernel_idx: int, mutated_arg_indices: tuple[int, ...], args: tuple[Any, ...]
) -> None:
    return None


@compiled_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def _mutation_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    mutated_arg_indices: tuple[int, ...],
    args: tuple[Any, ...],
) -> None:
    trace_user_kernel_wrapper(
        mode,
        compiled_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "mutated_arg_indices": mutated_arg_indices,
            "args": args,
        },
    )


@compiled_kernel_wrapper_mutation.py_functionalize_impl
def _mutation_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    mutated_arg_indices: tuple[int, ...],
    args: tuple[Any, ...],
) -> None:
    # Track view bases before redispatching so cloning can use them as graph inputs.
    tensor_bases = {}
    for i in mutated_arg_indices:
        arg = args[i]
        base = arg._base if isinstance(arg, Tensor) else None
        if base is not None:
            tensor_bases[i] = ctx.unwrap_tensors(base)

    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        outputs = compiled_kernel_wrapper_functional(
            kernel_idx,
            mutated_arg_indices,
            unwrapped_args,
            tensor_bases=tensor_bases,
        )

    for i, output in outputs.items():
        arg = args[i]
        ctx.replace(arg, output)
        ctx.mark_mutation_hidden_from_autograd(arg)
        ctx.commit_update(arg)
        ctx.sync(arg)


@compiled_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def _functional_dense(
    *,
    kernel_idx: int,
    mutated_arg_indices: tuple[int, ...],
    args: tuple[Any, ...],
    tensor_bases: dict[int, Tensor] | None = None,
) -> dict[int, Tensor]:
    clones = _functional_fake(
        kernel_idx=kernel_idx,
        mutated_arg_indices=mutated_arg_indices,
        args=args,
        tensor_bases=tensor_bases,
    )
    new_args = list(args)
    for i, clone in clones.items():
        new_args[i] = clone
    compiled_kernel_wrapper_mutation(kernel_idx, mutated_arg_indices, tuple(new_args))
    return clones


@register_fake(compiled_kernel_wrapper_functional, skip_cache=True)
def _functional_fake(
    *,
    kernel_idx: int,
    mutated_arg_indices: tuple[int, ...],
    args: tuple[Any, ...],
    tensor_bases: dict[int, Tensor] | None = None,
) -> dict[int, Tensor]:
    return {i: clone_mutated_arg(i, args[i], tensor_bases) for i in mutated_arg_indices}


@compiled_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def _functional_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    mutated_arg_indices: tuple[int, ...],
    args: tuple[Any, ...],
    tensor_bases: dict[int, Tensor] | None = None,
) -> dict[int, Tensor]:
    node_args: dict[str, Any] = {
        "kernel_idx": kernel_idx,
        "mutated_arg_indices": mutated_arg_indices,
        "args": args,
    }
    if tensor_bases:
        node_args["tensor_bases"] = tensor_bases
    return trace_user_kernel_wrapper(
        mode, compiled_kernel_wrapper_functional, node_args
    )


@compiled_kernel_wrapper_functional.py_functionalize_impl
def _functional_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    mutated_arg_indices: tuple[int, ...],
    args: tuple[Any, ...],
    tensor_bases: dict[int, Tensor] | None = None,
) -> dict[int, Tensor]:
    unwrapped_args = ctx.unwrap_tensors(args)
    unwrapped_bases = {
        i: ctx.unwrap_tensors(base) for i, base in (tensor_bases or {}).items()
    }
    with ctx.redispatch_to_next():
        outputs = compiled_kernel_wrapper_functional(
            kernel_idx,
            mutated_arg_indices,
            unwrapped_args,
            tensor_bases=unwrapped_bases,
        )
    wrapped = ctx.wrap_tensors(tuple(outputs.values()))
    return dict(zip(outputs, wrapped, strict=True))


register_user_kernel_hop_dispatch(
    compiled_kernel_wrapper_mutation, compiled_kernel_wrapper_functional
)

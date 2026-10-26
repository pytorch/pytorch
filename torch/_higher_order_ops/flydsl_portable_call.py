# mypy: allow-untyped-defs
from __future__ import annotations

import inspect
from typing import Any, cast, TYPE_CHECKING

import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._higher_order_ops.flydsl_program import (
    FlyDSLCallSpec,
    FlyDSLProgramSpec,
)
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.node import has_side_effect

if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


def _invoke(program_json: str, call_spec_json: str, args: tuple[Any, ...]) -> None:
    program = FlyDSLProgramSpec.from_json(program_json)
    launcher = program.resolve()
    compile_time = FlyDSLCallSpec.from_json(call_spec_json).decode()
    restored = list(args)
    for index, value in compile_time.items():
        restored[index] = value
    signature = inspect.signature(launcher.func)
    positional = []
    keyword = {}
    for parameter, value in zip(signature.parameters.values(), restored):
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword[parameter.name] = value
        else:
            raise TypeError("portable FlyDSL launchers cannot be variadic")
    launcher(*positional, **keyword)


class FlyDSLPortableMutation(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flydsl_portable_mutation", cacheable=True)

    def __call__(
        self,
        program_json: str,
        call_spec_json: str,
        args: tuple[Any, ...],
        mutated_arg_indices: tuple[int, ...],
    ) -> None:
        return super().__call__(
            program_json=program_json,
            call_spec_json=call_spec_json,
            args=args,
            mutated_arg_indices=mutated_arg_indices,
        )


class FlyDSLPortableFunctional(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flydsl_portable_functional", cacheable=True)

    def __call__(
        self,
        program_json: str,
        call_spec_json: str,
        args: tuple[Any, ...],
        mutated_arg_indices: tuple[int, ...],
    ) -> tuple[Tensor, ...]:
        return super().__call__(
            program_json=program_json,
            call_spec_json=call_spec_json,
            args=args,
            mutated_arg_indices=mutated_arg_indices,
        )


flydsl_portable_mutation = FlyDSLPortableMutation()
flydsl_portable_functional = FlyDSLPortableFunctional()
has_side_effect(flydsl_portable_mutation)


@flydsl_portable_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def flydsl_portable_mutation_dense(
    *,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    _invoke(program_json, call_spec_json, args)


@register_fake(flydsl_portable_mutation, skip_cache=True)
def flydsl_portable_mutation_fake(
    *,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    return None


def _trace(
    mode: ProxyTorchDispatchMode,
    op: HigherOrderOperator,
    kwargs: dict[str, Any],
) -> Any:
    with disable_proxy_modes_tracing():
        output = op(**kwargs)
    proxy_kwargs = pytree.tree_map(cast(Any, mode.tracer).unwrap_proxy, kwargs)
    proxy = mode.tracer.create_proxy(
        "call_function",
        op,
        (),
        proxy_kwargs,
        name=op.__name__ + "_proxy",
    )
    return track_tensor_tree(
        output,
        proxy,
        constant=None,
        tracer=mode.tracer,
    )


@flydsl_portable_mutation.py_impl(ProxyTorchDispatchMode)
def flydsl_portable_mutation_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    _trace(
        mode,
        flydsl_portable_mutation,
        {
            "program_json": program_json,
            "call_spec_json": call_spec_json,
            "args": args,
            "mutated_arg_indices": mutated_arg_indices,
        },
    )
    return None


@flydsl_portable_mutation.py_functionalize_impl
def flydsl_portable_mutation_functionalize(
    ctx: BaseFunctionalizeAPI,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    unwrapped = list(ctx.unwrap_tensors(args))
    for index in mutated_arg_indices:
        value = unwrapped[index]
        if not isinstance(value, Tensor):
            raise TypeError("mutated FlyDSL arguments must be tensors")
        unwrapped[index] = clone_preserve_strides(value)
    with ctx.redispatch_to_next():
        outputs = flydsl_portable_functional(
            program_json,
            call_spec_json,
            tuple(unwrapped),
            mutated_arg_indices,
        )
    for index, output in zip(mutated_arg_indices, outputs):
        original = args[index]
        ctx.replace(original, output)
        ctx.mark_mutation_hidden_from_autograd(original)
        ctx.commit_update(original)
        ctx.sync(original)


@flydsl_portable_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def flydsl_portable_functional_dense(
    *,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> tuple[Tensor, ...]:
    cloned = list(args)
    for index in mutated_arg_indices:
        value = cloned[index]
        if not isinstance(value, Tensor):
            raise TypeError("mutated FlyDSL arguments must be tensors")
        cloned[index] = clone_preserve_strides(value)
    flydsl_portable_mutation(
        program_json,
        call_spec_json,
        tuple(cloned),
        mutated_arg_indices,
    )
    return tuple(cloned[index] for index in mutated_arg_indices)


@register_fake(flydsl_portable_functional, skip_cache=True)
def flydsl_portable_functional_fake(
    *,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> tuple[Tensor, ...]:
    return tuple(clone_preserve_strides(args[index]) for index in mutated_arg_indices)


@flydsl_portable_functional.py_impl(ProxyTorchDispatchMode)
def flydsl_portable_functional_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> tuple[Tensor, ...]:
    return _trace(
        mode,
        flydsl_portable_functional,
        {
            "program_json": program_json,
            "call_spec_json": call_spec_json,
            "args": args,
            "mutated_arg_indices": mutated_arg_indices,
        },
    )


@flydsl_portable_functional.py_functionalize_impl
def flydsl_portable_functional_functionalize(
    ctx: BaseFunctionalizeAPI,
    program_json: str,
    call_spec_json: str,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> tuple[Tensor, ...]:
    unwrapped = tuple(ctx.unwrap_tensors(args))
    with ctx.redispatch_to_next():
        outputs = flydsl_portable_functional(
            program_json,
            call_spec_json,
            unwrapped,
            mutated_arg_indices,
        )
    return tuple(ctx.wrap_tensors(outputs))


for op in (flydsl_portable_mutation, flydsl_portable_functional):
    op.fallthrough(DispatchKey.PythonDispatcher)
    op.fallthrough(DispatchKey.ADInplaceOrView)
    op.fallthrough(DispatchKey.BackendSelect)

# mypy: allow-untyped-defs
from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


DispatchKey = cast(Any, torch._C).DispatchKey


@dataclass(frozen=True)
class FlyDSLLauncherRegistration:
    launcher: Any
    bound_self: Any
    signature: inspect.Signature
    mutated_arg_indices: tuple[int, ...]
    compile_time_arg_indices: frozenset[int]
    constexpr_arg_indices: frozenset[int]
    constexpr_value_signature: Callable[[Any], Any] | None
    stream_type: type[Any] | None
    jit_argument_type: type[Any] | None


class TraceableFlyDSLLauncher:
    def __init__(
        self,
        launcher: Any,
        mutated_arg_indices: tuple[int, ...],
        *,
        bound_self: Any = None,
        signature: inspect.Signature | None = None,
        compile_time_arg_indices: frozenset[int] = frozenset(),
        constexpr_arg_indices: frozenset[int] = frozenset(),
        constexpr_value_signature: Callable[[Any], Any] | None = None,
        stream_type: type[Any] | None = None,
        jit_argument_type: type[Any] | None = None,
    ) -> None:
        if signature is None:
            signature = inspect.signature(launcher.func)
        registration = FlyDSLLauncherRegistration(
            launcher,
            bound_self,
            signature,
            mutated_arg_indices,
            compile_time_arg_indices,
            constexpr_arg_indices,
            constexpr_value_signature,
            stream_type,
            jit_argument_type,
        )
        self.launcher_idx = flydsl_launcher_side_table.add_launcher(registration)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        registration = flydsl_launcher_side_table.get_registration(self.launcher_idx)
        bound = registration.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        runtime_args, call_spec_idx = partition_flydsl_launcher_arguments(
            registration,
            tuple(bound.arguments.values()),
        )
        flydsl_kernel_wrapper_mutation(
            self.launcher_idx,
            call_spec_idx,
            runtime_args,
            registration.mutated_arg_indices,
        )


class FlyDSLLauncherSideTable:
    def __init__(self) -> None:
        self.id_to_launcher: dict[int, FlyDSLLauncherRegistration] = {}
        self.launcher_to_id: dict[tuple[int, int | None, tuple[int, ...]], int] = {}
        self.id_to_call_spec: dict[int, dict[int, Any]] = {}
        self.call_spec_to_id: dict[tuple[Any, ...], int] = {}
        self.lock = threading.Lock()

    def add_launcher(self, registration: FlyDSLLauncherRegistration) -> int:
        key = (
            id(registration.launcher),
            (
                id(registration.bound_self)
                if registration.bound_self is not None
                else None
            ),
            registration.mutated_arg_indices,
        )
        with self.lock:
            if key in self.launcher_to_id:
                return self.launcher_to_id[key]
            idx = len(self.id_to_launcher)
            self.id_to_launcher[idx] = registration
            self.launcher_to_id[key] = idx
            return idx

    def get_registration(self, idx: int) -> FlyDSLLauncherRegistration:
        if idx not in self.id_to_launcher:
            raise AssertionError(f"FlyDSL launcher index {idx} was not registered")
        return self.id_to_launcher[idx]

    def get_launcher(self, idx: int) -> Any:
        return self.get_registration(idx).launcher

    def add_call_spec(
        self,
        constant_args: dict[int, Any],
        key: tuple[Any, ...] | None = None,
    ) -> int:
        if key is None:
            key = tuple(
                (idx, _constant_key(value)) for idx, value in constant_args.items()
            )
        with self.lock:
            if key in self.call_spec_to_id:
                return self.call_spec_to_id[key]
            idx = len(self.id_to_call_spec)
            self.id_to_call_spec[idx] = constant_args
            self.call_spec_to_id[key] = idx
            return idx

    def get_call_spec(self, idx: int) -> dict[int, Any]:
        if idx not in self.id_to_call_spec:
            raise AssertionError(f"FlyDSL call spec index {idx} was not registered")
        return self.id_to_call_spec[idx]

    def reset_table(self) -> None:
        with self.lock:
            self.id_to_launcher = {}
            self.launcher_to_id = {}
            self.id_to_call_spec = {}
            self.call_spec_to_id = {}


flydsl_launcher_side_table = FlyDSLLauncherSideTable()


def _constant_key(value: Any) -> tuple[Any, ...]:
    try:
        hash(value)
    except TypeError:
        return ("identity", type(value), id(value))
    return ("value", type(value), value)


def _register_flydsl_call_spec(
    constant_args: dict[int, Any],
    constexpr_indices: tuple[int, ...],
    constexpr_value_signature: Callable[[Any], Any] | None,
) -> int:
    constexpr_indices_set = set(constexpr_indices)
    key = tuple(
        (
            idx,
            (
                (
                    "constexpr",
                    (
                        constexpr_value_signature(value)
                        if constexpr_value_signature is not None
                        else _constant_key(value)
                    ),
                )
                if idx in constexpr_indices_set
                else ("type_parameter", value)
            ),
        )
        for idx, value in constant_args.items()
    )
    return flydsl_launcher_side_table.add_call_spec(constant_args, key)


def _validate_runtime_arguments(
    registration: FlyDSLLauncherRegistration,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    for idx, (parameter, value) in enumerate(
        zip(registration.signature.parameters.values(), args)
    ):
        if idx in registration.compile_time_arg_indices:
            continue
        if registration.stream_type is not None and isinstance(
            value, registration.stream_type
        ):
            raise TypeError(
                "wrap_flydsl does not accept explicit stream arguments; "
                "AOT launchers use PyTorch's current device stream"
            )
        if registration.jit_argument_type is not None and isinstance(
            value, registration.jit_argument_type
        ):
            raise TypeError(
                "wrap_flydsl runtime arguments must be graphable PyTorch values; "
                f"preconstructed FlyDSL JitArgument {type(value).__name__} is unsupported"
            )

    parameters = tuple(registration.signature.parameters.values())
    for idx in mutated_arg_indices:
        if not isinstance(args[idx], Tensor):
            raise TypeError(
                "wrap_flydsl mutated arguments must be tensors; "
                f"{parameters[idx].name!r} received {type(args[idx]).__name__}"
            )


def partition_flydsl_launcher_arguments(
    registration: FlyDSLLauncherRegistration,
    args: tuple[Any, ...],
) -> tuple[tuple[Any, ...], int]:
    signature = registration.signature
    parameters = tuple(signature.parameters.values())
    if len(args) != len(parameters):
        raise TypeError(
            f"FlyDSL launcher expected {len(parameters)} arguments, got {len(args)}"
        )
    runtime_args = list(args)
    constant_args = {}
    constexpr_indices = []
    for idx, value in enumerate(args):
        if idx in registration.constexpr_arg_indices:
            constexpr_indices.append(idx)
            constant_args[idx] = value
            runtime_args[idx] = None
        elif idx in registration.compile_time_arg_indices:
            constant_args[idx] = value
            runtime_args[idx] = None
    return tuple(runtime_args), _register_flydsl_call_spec(
        constant_args,
        tuple(constexpr_indices),
        registration.constexpr_value_signature,
    )


def restore_flydsl_launcher_arguments(
    args: tuple[Any, ...],
    call_spec_idx: int,
) -> tuple[Any, ...]:
    restored_args = list(args)
    for idx, value in flydsl_launcher_side_table.get_call_spec(call_spec_idx).items():
        restored_args[idx] = value
    return tuple(restored_args)


def split_flydsl_launcher_arguments(
    signature: inspect.Signature,
    args: tuple[Any, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    parameters = tuple(signature.parameters.values())
    if len(args) != len(parameters):
        raise TypeError(
            f"FlyDSL launcher expected {len(parameters)} arguments, got {len(args)}"
        )
    positional_args = []
    keyword_args = {}
    for parameter, value in zip(parameters, args):
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_args.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword_args[parameter.name] = value
        else:
            raise TypeError(
                "FlyDSL launchers with variadic parameters cannot be captured"
            )
    return tuple(positional_args), keyword_args


def invoke_flydsl_launcher(
    registration: FlyDSLLauncherRegistration,
    args: tuple[Any, ...],
) -> None:
    positional_args, keyword_args = split_flydsl_launcher_arguments(
        registration.signature,
        args,
    )
    if registration.bound_self is not None:
        positional_args = (registration.bound_self, *positional_args)
    registration.launcher(*positional_args, **keyword_args)


class FlyDSLKernelWrapperMutation(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flydsl_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self,
        launcher_idx: int,
        call_spec_idx: int,
        args: tuple[Any, ...],
        mutated_arg_indices: tuple[int, ...],
    ) -> None:
        return super().__call__(
            launcher_idx=launcher_idx,
            call_spec_idx=call_spec_idx,
            args=args,
            mutated_arg_indices=mutated_arg_indices,
        )


class FlyDSLKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flydsl_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self,
        launcher_idx: int,
        call_spec_idx: int,
        args: tuple[Any, ...],
        mutated_arg_indices: tuple[int, ...],
        tensors_to_clone: tuple[int, ...],
    ) -> tuple[Tensor, ...]:
        return super().__call__(
            launcher_idx=launcher_idx,
            call_spec_idx=call_spec_idx,
            args=args,
            mutated_arg_indices=mutated_arg_indices,
            tensors_to_clone=tensors_to_clone,
        )


flydsl_kernel_wrapper_mutation = FlyDSLKernelWrapperMutation()
flydsl_kernel_wrapper_functional = FlyDSLKernelWrapperFunctional()


@flydsl_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def flydsl_kernel_wrapper_mutation_dense(
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    registration = flydsl_launcher_side_table.get_registration(launcher_idx)
    full_args = restore_flydsl_launcher_arguments(args, call_spec_idx)
    _validate_runtime_arguments(registration, full_args, mutated_arg_indices)
    cuda_devices = {
        arg.device
        for arg in full_args
        if isinstance(arg, Tensor) and arg.device.type == "cuda"
    }
    for device in cuda_devices:
        if torch.cuda.current_stream(device) != torch.cuda.default_stream(device):
            raise RuntimeError(
                "eager wrap_flydsl launches require the default device stream; "
                "AOTInductor launchers use PyTorch's current device stream"
            )
    invoke_flydsl_launcher(
        registration,
        full_args,
    )


@register_fake(flydsl_kernel_wrapper_mutation, skip_cache=True)
def flydsl_kernel_wrapper_mutation_fake(
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    return None


@flydsl_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def flydsl_kernel_wrapper_mutation_meta(
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    return None


def _trace_flydsl_wrapper(
    mode: ProxyTorchDispatchMode,
    op: HigherOrderOperator,
    node_args: dict[str, Any],
) -> Any:
    with disable_proxy_modes_tracing():
        output = op(**node_args)
    proxy_args = pytree.tree_map(cast(Any, mode.tracer).unwrap_proxy, node_args)
    output_proxy = mode.tracer.create_proxy(
        "call_function",
        op,
        (),
        proxy_args,
        name=op.__name__ + "_proxy",
    )
    return track_tensor_tree(
        output,
        output_proxy,
        constant=None,
        tracer=mode.tracer,
    )


@flydsl_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def flydsl_kernel_wrapper_mutation_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    _trace_flydsl_wrapper(
        mode,
        flydsl_kernel_wrapper_mutation,
        {
            "launcher_idx": launcher_idx,
            "call_spec_idx": call_spec_idx,
            "args": args,
            "mutated_arg_indices": mutated_arg_indices,
        },
    )
    return None


@flydsl_kernel_wrapper_mutation.py_functionalize_impl
def flydsl_kernel_wrapper_mutation_functionalize(
    ctx: BaseFunctionalizeAPI,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
) -> None:
    unwrapped_args = tuple(ctx.unwrap_tensors(args))
    with ctx.redispatch_to_next():
        outputs = flydsl_kernel_wrapper_functional(
            launcher_idx,
            call_spec_idx,
            unwrapped_args,
            mutated_arg_indices,
            tuple(range(len(mutated_arg_indices))),
        )

    for arg_idx, output in zip(mutated_arg_indices, outputs):
        input_arg = args[arg_idx]
        if not isinstance(input_arg, Tensor):
            raise AssertionError(
                f"Expected FlyDSL mutated argument {arg_idx} to be a Tensor"
            )
        ctx.replace(input_arg, output)
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)


@flydsl_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def flydsl_kernel_wrapper_functional_dense(
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
    tensors_to_clone: tuple[int, ...],
) -> tuple[Tensor, ...]:
    _validate_clone_aliases(args, mutated_arg_indices, tensors_to_clone)
    cloned_args = list(args)
    for output_idx in tensors_to_clone:
        arg_idx = mutated_arg_indices[output_idx]
        arg = cloned_args[arg_idx]
        if not isinstance(arg, Tensor):
            raise AssertionError(
                f"Expected FlyDSL mutated argument {arg_idx} to be a Tensor"
            )
        cloned_args[arg_idx] = clone_preserve_strides(arg)
    flydsl_kernel_wrapper_mutation(
        launcher_idx,
        call_spec_idx,
        tuple(cloned_args),
        mutated_arg_indices,
    )
    return tuple(cloned_args[idx] for idx in mutated_arg_indices)


@register_fake(flydsl_kernel_wrapper_functional, skip_cache=True)
def flydsl_kernel_wrapper_functional_fake(
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
    tensors_to_clone: tuple[int, ...],
) -> tuple[Tensor, ...]:
    _validate_clone_aliases(args, mutated_arg_indices, tensors_to_clone)
    clone_indices = set(tensors_to_clone)
    return tuple(
        (
            clone_preserve_strides(args[arg_idx])
            if output_idx in clone_indices
            else args[arg_idx]
        )
        for output_idx, arg_idx in enumerate(mutated_arg_indices)
    )


@flydsl_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def flydsl_kernel_wrapper_functional_proxy(
    mode: ProxyTorchDispatchMode,
    *,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
    tensors_to_clone: tuple[int, ...],
) -> tuple[Tensor, ...]:
    return _trace_flydsl_wrapper(
        mode,
        flydsl_kernel_wrapper_functional,
        {
            "launcher_idx": launcher_idx,
            "call_spec_idx": call_spec_idx,
            "args": args,
            "mutated_arg_indices": mutated_arg_indices,
            "tensors_to_clone": tensors_to_clone,
        },
    )


@flydsl_kernel_wrapper_functional.py_functionalize_impl
def flydsl_kernel_wrapper_functional_functionalize(
    ctx: BaseFunctionalizeAPI,
    launcher_idx: int,
    call_spec_idx: int,
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
    tensors_to_clone: tuple[int, ...],
) -> tuple[Tensor, ...]:
    unwrapped_args = tuple(ctx.unwrap_tensors(args))
    with ctx.redispatch_to_next():
        outputs = flydsl_kernel_wrapper_functional(
            launcher_idx,
            call_spec_idx,
            unwrapped_args,
            mutated_arg_indices,
            tensors_to_clone,
        )
    return tuple(ctx.wrap_tensors(outputs))


def _validate_clone_aliases(
    args: tuple[Any, ...],
    mutated_arg_indices: tuple[int, ...],
    tensors_to_clone: tuple[int, ...],
) -> None:
    cloned_arg_indices = {
        mutated_arg_indices[output_idx] for output_idx in tensors_to_clone
    }
    tensor_args = [
        (idx, arg) for idx, arg in enumerate(args) if isinstance(arg, Tensor)
    ]
    for position, (arg_idx, arg) in enumerate(tensor_args):
        for other_idx, other in tensor_args[position + 1 :]:
            if (
                arg_idx in cloned_arg_indices or other_idx in cloned_arg_indices
            ) and cast(Any, torch._C)._is_alias_of(arg, other):
                raise RuntimeError(
                    "FlyDSL functionalization does not support cloning aliased "
                    f"launcher arguments {arg_idx} and {other_idx}"
                )


for op in (flydsl_kernel_wrapper_mutation, flydsl_kernel_wrapper_functional):
    op.fallthrough(DispatchKey.PythonDispatcher)
    op.fallthrough(DispatchKey.PythonTLSSnapshot)
    op.fallthrough(DispatchKey.ADInplaceOrView)
    op.fallthrough(DispatchKey.BackendSelect)
    op.fallthrough(DispatchKey.AutocastCPU)
    op.fallthrough(DispatchKey.AutocastCUDA)
    op.fallthrough(DispatchKey.AutogradCPU)
    op.fallthrough(DispatchKey.AutogradCUDA)

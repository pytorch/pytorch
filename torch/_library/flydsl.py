import inspect
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

from torch.utils._exposed_in import exposed_in

from .custom_ops import custom_op, CustomOpDef
from .infer_schema import infer_schema


@exposed_in("torch.library")
def flydsl_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    schema: str | None = None,
) -> Callable:
    """Create a custom operator backed by one or more wrapped FlyDSL launchers.

    Use this boundary when a FlyDSL-backed operation contains Python control flow
    that should remain opaque to frontend tracing. The function body may contain
    PyTorch-understood operations and launchers returned by :func:`wrap_flydsl`.
    It must otherwise remain traceable when its implementation is decomposed for
    compilation and must not rely on unsupported external Python side effects.

    Args:
        name: A stable operator name in ``"namespace::name"`` form.
        mutates_args: Names of arguments mutated by the outer operator, or
            ``"unknown"`` to conservatively mark all inputs as mutated.
        schema: An optional operator schema. By default it is inferred from the
            function annotations.

    Example::

        wrapped_launcher = torch.library.wrap_flydsl(launcher, mutates_args={"out"})


        @torch.library.flydsl_op("mylib::add_one", mutates_args=())
        def add_one(inp: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(inp)
            wrapped_launcher(out=out, inp=inp, rows=inp.numel())
            return out
    """

    def dec(fn: Callable[..., object]) -> CustomOpDef:
        result = custom_op(
            name,
            fn,
            mutates_args=mutates_args,
            schema=(
                schema
                if schema is not None
                else infer_schema(fn, mutates_args=mutates_args)
            ),
        )
        result.register_fake(fn)

        from .._subclasses.functional_tensor import FunctionalTensorMode

        def functional_decomp(mode, op, types, args, kwargs):
            import torch._subclasses

            unrecognized_types = [
                typ
                for typ in types
                if not issubclass(typ, torch._subclasses.FakeTensor)
                and typ
                not in (
                    torch.Tensor,
                    torch._subclasses.functional_tensor.FunctionalTensor,
                )
            ]
            if unrecognized_types:
                return NotImplemented
            with mode:
                return fn(*args, **kwargs)

        result.register_torch_dispatch(FunctionalTensorMode, functional_decomp)
        return result

    if fn is None:
        return dec
    return dec(fn)


@exposed_in("torch.library")
def wrap_flydsl(
    launcher: Callable[..., Any],
    /,
    *,
    mutates_args: Iterable[str],
) -> Any:
    """Wrap a FlyDSL ``@jit`` launcher for dispatcher-based tracing.

    The launcher must write results into explicit tensor arguments and return
    ``None``. AOTInductor executes a wrapped launcher on PyTorch's current
    device stream. Direct eager calls require the default device stream;
    launchers with an explicit FlyDSL ``Stream`` parameter are unsupported.
    ``mutates_args`` names the tensor arguments written by the launcher.
    Runtime arguments must be graphable PyTorch values rather than
    preconstructed FlyDSL ``JitArgument`` objects.

    Variadic launcher parameters are not supported.
    """
    from torch._dynamo.decorators import allow_in_graph, assume_constant_result
    from torch._higher_order_ops.flydsl_kernel_wrap import (
        _register_flydsl_call_spec,
        flydsl_kernel_wrapper_mutation,
        TraceableFlyDSLLauncher,
    )
    from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available

    if not runtime_available():
        raise RuntimeError(
            "wrap_flydsl requires a supported optional `flydsl` runtime "
            "on a ROCm-enabled build"
        )

    from flydsl.compiler import jit_argument, jit_function, protocol
    from flydsl.expr import typing as flydsl_typing

    JitFunction = jit_function.JitFunction
    bound_self = None
    jit_launcher = launcher
    if isinstance(launcher, partial):
        bound_call = launcher.func
        candidate = getattr(bound_call, "__self__", None)
        if (
            isinstance(candidate, JitFunction)
            and getattr(bound_call, "__func__", None) is JitFunction.__call__
            and len(launcher.args) == 1
            and not launcher.keywords
        ):
            jit_launcher = candidate
            bound_self = launcher.args[0]

    if not isinstance(jit_launcher, JitFunction):
        raise RuntimeError(
            "wrap_flydsl only works on functions annotated with flydsl.compiler.jit"
        )
    signature = jit_argument.resolve_signature(jit_launcher.func)
    parameters = tuple(signature.parameters.values())
    has_self = bool(parameters) and parameters[0].name == "self"
    if has_self:
        if bound_self is None:
            raise TypeError("FlyDSL JIT methods must be wrapped from a bound instance")
        signature = signature.replace(parameters=parameters[1:])
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    ):
        raise TypeError("FlyDSL launchers with variadic parameters cannot be wrapped")
    if any(
        getattr(parameter.annotation, "_is_stream_param", False)
        for parameter in signature.parameters.values()
    ):
        raise TypeError(
            "FlyDSL launchers with explicit Stream parameters cannot be wrapped; "
            "AOT launchers use PyTorch's current device stream"
        )

    compile_time_arg_indices = frozenset(
        idx
        for idx, parameter in enumerate(signature.parameters.values())
        if parameter.annotation is not inspect.Parameter.empty
        and (
            flydsl_typing.Constexpr.is_constexpr_annotation(parameter.annotation)
            or jit_argument.is_type_param_annotation(parameter.annotation)
        )
    )
    constexpr_arg_indices = frozenset(
        idx
        for idx, parameter in enumerate(signature.parameters.values())
        if parameter.annotation is not inspect.Parameter.empty
        and flydsl_typing.Constexpr.is_constexpr_annotation(parameter.annotation)
    )

    mutations = frozenset(mutates_args)
    unknown = mutations.difference(signature.parameters)
    if unknown:
        raise ValueError(
            f"FlyDSL mutated arguments are not launcher parameters: {sorted(unknown)}"
        )
    parameter_names = tuple(signature.parameters)
    compile_time_parameters = {parameter_names[idx] for idx in compile_time_arg_indices}
    invalid_mutations = mutations.intersection(compile_time_parameters)
    if invalid_mutations:
        raise ValueError(
            "FlyDSL compile-time arguments cannot be mutated: "
            f"{sorted(invalid_mutations)}"
        )
    non_tensor_mutations = {
        name
        for name in mutations
        if not (
            isinstance(signature.parameters[name].annotation, type)
            and issubclass(
                signature.parameters[name].annotation,
                flydsl_typing.Tensor,
            )
        )
    }
    if non_tensor_mutations:
        raise TypeError(
            "FlyDSL mutated arguments must have the flydsl.expr.Tensor "
            f"annotation: {sorted(non_tensor_mutations)}"
        )
    mutated_arg_indices = tuple(
        idx for idx, name in enumerate(parameter_names) if name in mutations
    )
    assume_constant_result(_register_flydsl_call_spec)
    allow_in_graph(flydsl_kernel_wrapper_mutation)
    return TraceableFlyDSLLauncher(
        jit_launcher,
        mutated_arg_indices,
        bound_self=bound_self,
        signature=signature,
        compile_time_arg_indices=compile_time_arg_indices,
        constexpr_arg_indices=constexpr_arg_indices,
        constexpr_value_signature=flydsl_typing.Constexpr.value_signature,
        stream_type=flydsl_typing.Stream,
        jit_argument_type=protocol.JitArgument,
    )

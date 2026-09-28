import inspect
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

from torch.utils._exposed_in import exposed_in


@exposed_in("torch.library")
def wrap_flydsl(
    launcher: Callable[..., Any],
    /,
    *,
    mutates_args: str | Iterable[str],
) -> Callable[..., None]:
    """Wrap a FlyDSL ``@jit`` launcher for dispatcher-based tracing.

    The launcher must write results into explicit tensor arguments and return
    ``None``. AOTInductor executes a wrapped launcher on PyTorch's current
    device stream. Direct eager calls require the default device stream;
    Defaulted FlyDSL ``Stream`` parameters are omitted so the launcher uses the
    current device stream; callers cannot pass an explicit stream value.
    ``mutates_args`` names the tensor arguments written by the launcher.
    Runtime arguments must be graphable PyTorch values rather than
    preconstructed FlyDSL ``JitArgument`` objects.

    Variadic launcher parameters are not supported.

    Args:
        launcher: A function decorated with ``flydsl.compiler.jit``.
        mutates_args: The name, or names, of tensor arguments mutated by the
            launcher.

    Returns:
        A callable launcher that can be captured by PyTorch compilation APIs.

    Example::

        captured_launcher = torch.library.wrap_flydsl(
            launcher,
            mutates_args="out",
        )
        captured_launcher(out, inp, inp.numel())
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
    if bound_self is not None:
        if not parameters:
            raise TypeError("Bound FlyDSL JIT methods must declare a receiver")
        if parameters[0].name != "self":
            raise TypeError("Bound FlyDSL JIT methods must name their receiver 'self'")
        signature = signature.replace(parameters=parameters[1:])
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    ):
        raise TypeError("FlyDSL launchers with variadic parameters cannot be wrapped")
    stream_parameters = tuple(
        parameter
        for parameter in signature.parameters.values()
        if getattr(parameter.annotation, "_is_stream_param", False)
    )
    required_stream_parameters = tuple(
        parameter.name
        for parameter in stream_parameters
        if parameter.default is inspect.Parameter.empty
    )
    if required_stream_parameters:
        raise TypeError(
            "FlyDSL launchers with required Stream parameters cannot be wrapped; "
            "default the stream to use PyTorch's current device stream: "
            f"{list(required_stream_parameters)}"
        )
    if len(stream_parameters) > 1:
        raise TypeError("FlyDSL launchers may declare at most one Stream parameter")
    stream_parameter = (
        (
            tuple(signature.parameters).index(stream_parameters[0].name),
            stream_parameters[0],
        )
        if stream_parameters
        else None
    )
    signature = signature.replace(
        parameters=[
            parameter
            for parameter in signature.parameters.values()
            if parameter not in stream_parameters
        ]
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

    mutations = frozenset(
        (mutates_args,) if isinstance(mutates_args, str) else mutates_args
    )
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
        stream_parameter=stream_parameter,
        compile_time_arg_indices=compile_time_arg_indices,
        constexpr_arg_indices=constexpr_arg_indices,
        constexpr_value_signature=flydsl_typing.Constexpr.value_signature,
        stream_type=flydsl_typing.Stream,
        jit_argument_type=protocol.JitArgument,
    )

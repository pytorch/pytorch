import inspect
from collections.abc import Callable, Iterable
from typing import Any

from torch.utils._exposed_in import exposed_in


@exposed_in("torch.library")
def wrap_flydsl(
    launcher: Callable[..., Any],
    /,
    *,
    mutates_args: str | Iterable[str],
) -> Callable[..., None]:
    """Wrap a module-level FlyDSL launcher with a portable graph identity."""
    from torch._dynamo.decorators import allow_in_graph
    from torch._higher_order_ops.flydsl_portable_call import (
        flydsl_portable_mutation,
    )
    from torch._higher_order_ops.flydsl_program import (
        FlyDSLCallSpec,
        FlyDSLProgramSpec,
    )
    from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available

    if not runtime_available():
        raise RuntimeError("wrap_flydsl requires a supported FlyDSL runtime")

    from flydsl.compiler import jit_argument, jit_function
    from flydsl.expr import typing as flydsl_typing

    if not isinstance(launcher, jit_function.JitFunction):
        raise TypeError("wrap_flydsl expects a function decorated with @flyc.jit")
    signature = inspect.signature(launcher.func)
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    ):
        raise TypeError("portable FlyDSL launchers cannot be variadic")

    mutations = frozenset(
        (mutates_args,) if isinstance(mutates_args, str) else mutates_args
    )
    unknown = mutations.difference(signature.parameters)
    if unknown:
        raise ValueError(
            f"FlyDSL mutated arguments are not parameters: {sorted(unknown)}"
        )
    names = tuple(signature.parameters)
    mutated_indices = tuple(
        index for index, name in enumerate(names) if name in mutations
    )
    compile_time_indices = frozenset(
        index
        for index, parameter in enumerate(signature.parameters.values())
        if parameter.annotation is not inspect.Parameter.empty
        and (
            flydsl_typing.Constexpr.is_constexpr_annotation(parameter.annotation)
            or jit_argument.is_type_param_annotation(parameter.annotation)
        )
    )
    program = FlyDSLProgramSpec.from_launcher(
        launcher,
        mutated_arg_names=tuple(mutations),
    )
    program_json = program.to_json()
    allow_in_graph(flydsl_portable_mutation)

    class PortableFlyDSLLauncher:
        __name__ = launcher.func.__name__

        def __call__(self, *args, **kwargs) -> None:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = list(bound.arguments.values())
            import torch

            tensor_args = [
                (index, value)
                for index, value in enumerate(values)
                if isinstance(value, torch.Tensor)
            ]
            for mutated_index in mutated_indices:
                mutated = values[mutated_index]
                if not isinstance(mutated, torch.Tensor):
                    raise TypeError("mutated FlyDSL arguments must be tensors")
                if mutated._base is not None:
                    raise RuntimeError(
                        "portable FlyDSL functionalization does not support "
                        "mutated tensor views"
                    )
                for other_index, other in tensor_args:
                    if other_index != mutated_index and torch._C._is_alias_of(
                        mutated, other
                    ):
                        raise RuntimeError(
                            "portable FlyDSL functionalization does not support "
                            f"aliased arguments {mutated_index} and {other_index}"
                        )
            compile_time = {index: values[index] for index in compile_time_indices}
            for index in compile_time_indices:
                values[index] = None
            flydsl_portable_mutation(
                program_json,
                FlyDSLCallSpec.from_values(compile_time).to_json(),
                tuple(values),
                mutated_indices,
            )

    return PortableFlyDSLLauncher()

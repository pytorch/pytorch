import inspect
from collections.abc import Callable, Collection
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.export.c_header_generator import CHeaderGenerator


def export_args(
    fn: Callable[..., Any],
    *tensors: cute.Tensor,
    prefix: str,
    read_only: Collection[str] = (),
) -> dict[str, Any]:
    r"""Derive fake scalar arguments and AOT metadata from the callable signature."""
    parameters = list(inspect.signature(fn, eval_str=True).parameters.values())
    args = [*tensors]
    for param in parameters[len(tensors) : -1]:
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"AOT scalar argument {param.name!r} requires a type annotation"
            )
        args.append(param.annotation(1))
    args.append(cute.runtime.make_fake_stream())
    tensor_args, scalars = [], []
    for param, arg in zip(parameters, args, strict=True):
        name = param.name
        if isinstance(arg, cute.Tensor):
            sizes = [i for i, d in enumerate(arg.shape) if not isinstance(d, int)]
            strides = [i for i, d in enumerate(arg.stride) if not isinstance(d, int)]
            tensor_args.append(
                dict(
                    name=name,
                    dynamic_sizes=sizes,
                    dynamic_strides=strides,
                    read_only=name in read_only,
                )
            )
        elif isinstance(arg, cutlass.Numeric):
            ctype = CHeaderGenerator.numeric_to_c_type[type(arg)].strip()
            scalars.append(dict(name=name, ctype=ctype))
    return dict(
        kind="cutedsl",
        prefix=prefix,
        fn=fn,
        fake_args=list(args),
        tensor_args=tensor_args,
        scalar_args=scalars,
    )

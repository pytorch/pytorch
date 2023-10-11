import functools

import torch
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403

__all__: List[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}


def check_schema(schema_str: str, func, *args, **kwargs) -> None:
    named_arg_types = schema_str.split(", ")
    num_optional_args = sum([x.endswith("?") for x in named_arg_types])
    min_args = len(named_arg_types) - num_optional_args

    if not (len(args) >= min_args and len(args) <= len(named_arg_types)):
        raise ValueError(
            f"NestedTensor {func.__name__}({schema_str}): expected at least {min_args} "
            f"arguments and at most {len(named_arg_types)} arguments, but got: "
            f"{len(args)} arguments"
        )

    arg_type_check_fns = {
        "t": lambda x: isinstance(x, torch.Tensor) and not isinstance(x, NestedTensor),
        "jt": lambda x: isinstance(x, NestedTensor),
        "any": lambda x: True,
    }
    for i, named_arg_type in enumerate(named_arg_types):
        name, arg_type = named_arg_type.split(": ")
        is_optional = arg_type.endswith("?")
        normalized_arg_type = arg_type[:-1] if is_optional else arg_type
        if normalized_arg_type not in arg_type_check_fns.keys():
            raise AssertionError(f"Unknown arg type: {normalized_arg_type}")

        if i >= len(args):
            if not is_optional:
                raise ValueError(
                    f"NestedTensor {func.__name__}({schema_str}) "
                    f"missing required argument: {name}"
                )
            continue

        if not arg_type_check_fns[normalized_arg_type](args[i]):
            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): {name} should be of "
                f"type {arg_type}, but got: {type(args[i])}"
            )


def check_ragged_dim_same(
    func, a: NestedTensor, a_name: str, b: NestedTensor, b_name: str
) -> None:
    # Calling into .shape here
    assert len(a._size) == 3, "NestedTensor must be [B, *, D]"
    if a._size[1] != b._size[1]:
        raise RuntimeError(
            f"NestedTensor {func.__name__}: expected {a_name} and {b_name} to have the "
            "same exact offsets tensor."
        )


def register_func(tables, aten_ops, schema_str):
    if not isinstance(aten_ops, list):
        aten_ops = [aten_ops]
    if not isinstance(tables, list):
        tables = [tables]

    def wrapper(func):
        for aten_op in aten_ops:

            def get_inner(aten_op):
                def inner(*args, **kwargs):
                    check_schema(schema_str, func, *args, **kwargs)
                    return func(aten_op, *args, **kwargs)

                return inner

            for table in tables:
                table[aten_op] = get_inner(aten_op)

    return wrapper


register_jagged_func = functools.partial(register_func, JAGGED_OPS_TABLE)


def lookup_jagged(func, *args, **kwargs) -> Optional[Callable]:
    if torch.Tag.pointwise in func.tags:
        # Assume there aren't additional tensors that aren't the "unary/binary" args
        num_tensor_args = sum([isinstance(x, torch.Tensor) for x in args])
        if num_tensor_args == 1:
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema("lhs: jt, rhs: jt", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)
        else:
            return None
    return JAGGED_OPS_TABLE.get(func, None)


def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
        "ragged_size": arg._size[arg._ragged_idx],
    }
    return kwargs


def jagged_unary_pointwise(func, *args, **kwargs):
    return NestedTensor(func(args[0].values(), **kwargs), **extract_kwargs(args[0]))


def jagged_binary_pointwise(func, *args, **kwargs):
    check_ragged_dim_same(func, args[0], "lhs", args[1], "rhs")
    return NestedTensor(
        func(args[0].values(), args[1].values(), **kwargs), **extract_kwargs(args[0])
    )


@register_jagged_func(
    [
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.dim.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_storage_offset.default,
    ],
    "self: jt",
)
def tensor_attr_supported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.is_non_overlapping_and_dense.default:
        return False

    if func == torch.ops.aten.sym_size.default:
        return args[0]._size

    if func == torch.ops.aten.dim.default:
        return 3

    if func == torch.ops.aten.sym_numel.default:
        return args[0].values().numel()

    if func == torch.ops.aten.sym_stride.default:
        return args[0]._strides

    if func == torch.ops.aten.sym_storage_offset.default:
        return 0


@register_jagged_func(
    [
        torch.ops.aten.size.default,
        torch.ops.aten.is_contiguous.default,
        torch.ops.aten.is_contiguous.memory_format,
    ],
    "self: jt, memory_format: any?",
)
def tensor_attr_unsupported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.size.default:
        raise RuntimeError(
            "NestedTensors does not support directly calling torch.ops.aten.size "
            "please use `nested_tensor.size()` instead."
        )

    raise RuntimeError(
        "NestedTensors do not support directly querying strides, "
        "storage_offset, or contiguity."
    )


@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    values = torch.mm(args[0].values(), args[1])
    if len(args) == 3:
        values += args[2]
    return NestedTensor(values, **extract_kwargs(args[0]))


@register_jagged_func(
    torch.ops.aten.linear_backward.default,
    "self: jt, grad_output: jt, weight: t, output_mask: any",
)
def linear_backward_default(func, *args, **kwargs):
    check_ragged_dim_same(func, args[0], "self", args[1], "grad_output")
    ds = NestedTensor(torch.mm(args[1].values(), args[2].T), **extract_kwargs(args[1]))
    dw = torch.mm(args[0].values().T, args[1].values())
    db = None  # NYI: gradient for bias, need to reduce over ragged dim
    return (ds, dw, db)

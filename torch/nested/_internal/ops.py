import functools

import torch
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function

__all__: List[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}

def _wrap_jagged_dim(ndim, dim, op_name):
    import torch._prims_common
    from torch._prims_common import canonicalize_dims
    wrapped = canonicalize_dims(ndim, dim)
    if wrapped < 2:
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on dim=0 or dim=1")
    return wrapped - 1


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
    dispatch_func = JAGGED_OPS_TABLE.get(func, None)
    if dispatch_func is not None:
        return dispatch_func

    # Handle pointwise fallbacks
    if torch.Tag.pointwise in func.tags:
        # Assume there aren't additional tensors that aren't the "unary/binary" args
        num_tensor_args = sum([isinstance(x, torch.Tensor) for x in args])
        if num_tensor_args == 1:
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema("lhs: jt, rhs: any", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)

    return None

def jagged_unary_pointwise(func, *args, **kwargs):
    return NestedTensor(func(args[0]._values, *args[1:], **kwargs), args[0]._offsets)

def jagged_binary_pointwise(func, *args, **kwargs):
    a, b = args[0], args[1]
    assert isinstance(a, NestedTensor)
    if isinstance(b, NestedTensor):
        check_ragged_dim_same(func, args[0], "lhs", args[1], "rhs")
        b = args[1]._values
    else:
        # TODO: Verify this more and handle the a.dim() == b.dim() case specially if needed
        if a.dim() <= b.dim():
            # need to use offsets to broadcast across batch dim properly
            # NB: inefficient fallback here; Triton codegen can help this
            assert a.shape[0] == b.shape[0]
            outputs = []
            for a_comp, b_comp in zip(a.unbind(), b.unbind()):
                outputs.append(func(a_comp, b_comp, **kwargs))
            new_values = torch.cat(outputs, dim=0)
            return NestedTensor(new_values, a._offsets)
    return NestedTensor(func(a._values, b, **kwargs), a._offsets)

@register_jagged_func([
    torch.ops.aten.is_non_overlapping_and_dense.default,
    torch.ops.aten.sym_size.default,
    torch.ops.aten.dim.default,
    torch.ops.aten.sym_numel.default,
], "self: jt")
def tensor_attr_supported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.is_non_overlapping_and_dense.default:
        return False

    if func == torch.ops.aten.sym_size.default:
        return args[0]._size

    if func == torch.ops.aten.dim.default:
        return len(args[0]._size)

    if func == torch.ops.aten.sym_numel.default:
        return args[0]._values.numel()

@register_jagged_func([
    torch.ops.aten.size.default,
    torch.ops.aten.sym_stride.default,
    torch.ops.aten.is_contiguous.default,
    torch.ops.aten.is_contiguous.memory_format,
    torch.ops.aten.sym_storage_offset.default,
], "self: jt, memory_format: any?")
def tensor_attr_unsupported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.size.default:
        raise RuntimeError(
            "NestedTensors does not support directly calling torch.ops.aten.size "
            "please use `nested_tensor.size()` instead.")

    raise RuntimeError(
        "NestedTensors do not support directly querying strides, "
        "storage_offset, or contiguity.")

@register_jagged_func(torch.ops.aten.linear.default,
                      "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    new_values = torch.mm(args[0]._values, args[1])
    if len(args) == 3:
        new_values += args[2]
    return NestedTensor(new_values, args[0]._offsets)

@register_jagged_func(torch.ops.aten.linear_backward.default,
                      "self: jt, grad_output: jt, weight: t, output_mask: any")
def linear_backward_default(func, *args, **kwargs):
    check_ragged_dim_same(func, args[0], "self", args[1], "grad_output")
    ds = NestedTensor(torch.mm(args[1]._values, args[2].T), args[1]._offsets)
    dw = torch.mm(args[0]._values.T, args[1]._values)
    db = None  # NYI: gradient for bias, need to reduce over ragged dim
    return (ds, dw, db)

@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt")
def to_copy_default(func, *args, **kwargs):
    return NestedTensor(args[0]._values.to(
        device=kwargs['device'],
        dtype=kwargs['dtype'],
        non_blocking=kwargs['non_blocking'],
        copy=True
    ), args[0]._offsets)

register_jagged_func([
    torch.ops.aten.ones_like.default,
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.randn_like.default,
], "self: jt")(jagged_unary_pointwise)

@register_jagged_func(torch.ops.aten.prod.dim_int, "self: jt, dim: any, keepdim: any?")
def prod_dim_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # TODO: Figure out how to handle this better
    # keep_dim is required to keep it in jagged format
    if not new_kwargs['keepdim']:
        raise RuntimeError("prod(): keepdim=True must be set for NestedTensor")
    dim = new_kwargs['dim']
    new_kwargs['dim'] = _wrap_jagged_dim(len(inp.shape), dim, "prod")
    if new_kwargs['dim'] == 0:
        raise RuntimeError("prod(): not supported for NestedTensor on dim=0")

    return NestedTensor(func(inp._values, **new_kwargs), args[0]._offsets)

@register_jagged_func(torch.ops.aten.unbind.int, "self: jt, dim: any?")
def unbind_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dim = new_kwargs['dim']
    if dim != 0:
        raise RuntimeError("unbind(): only supported for NestedTensor on dim=0")

    inp = new_kwargs.pop("input")
    values = inp._values
    offsets = inp.offsets()

    views = []
    start = 0
    for length in offsets.diff().cpu().tolist():
        views.append(inp._values[start:start+length, ...])
        start += length

    return tuple(views)

@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values
    offsets = inp.offsets

    # Account for collapsed jagged dim
    dim = new_kwargs['dim']
    new_kwargs['dim'] = _wrap_jagged_dim(len(inp.shape) + 1, dim, "unsqueeze")
    return NestedTensor(func(values, **new_kwargs), inp._offsets)

@register_jagged_func(torch.ops.aten.cat.default, "tensors: any, dim: any")
def cat_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    tensors = new_kwargs.pop("tensors")

    # Convert any non-nested to nested
    nested = [t for t in tensors if t.is_nested]
    assert len(nested) > 0
    first = nested[0]
    tensors = [t if t.is_nested else t.expand_as(first) for t in tensors]

    # Account for collapsed jagged dim
    dim = new_kwargs['dim']
    new_kwargs['dim'] = _wrap_jagged_dim(len(first.shape), dim, "cat")

    return NestedTensor(func([t._values for t in tensors], **new_kwargs),
                        tensors[0]._offsets)

@register_jagged_func(torch.ops.aten.matmul.default, "self: jt, other: any")
def matmul_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")
    if (not inp.is_nested) or other.is_nested:
        raise RuntimeError("matmul(): only supported input pattern is (nested, non-nested)")
    return NestedTensor(func(inp._values, other, **new_kwargs), inp._offsets)

@register_jagged_func(torch.ops.aten.expand.default, "self: jt, size: any, implicit: any?")
def expand_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs['size']

    assert (not "implicit" in new_kwargs) or (not new_kwargs.pop("implicit"))
    if list(size[:2]) != list(inp.shape[:2]):
        raise RuntimeError("expand(): cannot expand if ragged dims don't match")

    expand_arg = [-1, *size[2:]]
    return NestedTensor(func(inp._values, expand_arg), inp._offsets)

@register_jagged_func(torch.ops.aten.expand_as.default, "self: t, other: jt")
def expand_as_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    return NestedTensor(func(inp, other._values), other._offsets)

@register_jagged_func(torch.ops.aten.where.self, "condition: jt, self: jt, other: jt")
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    condition = new_kwargs.pop("condition")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    assert (condition.shape == other.shape == inp.shape)

    return NestedTensor(func(condition._values, inp._values, other._values, **new_kwargs),
                        condition._offsets)

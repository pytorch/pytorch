import functools

import torch

from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function

__all__: List[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}


def _wrap_jagged_dim(ndim, dim, op_name):
    from torch._prims_common import canonicalize_dims

    wrapped = canonicalize_dims(ndim, dim)
    if wrapped < 2:
        raise RuntimeError(
            f"{op_name}(): not supported for NestedTensor on dim=0 or dim=1"
        )
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
        "jt": lambda x: isinstance(x, NestedTensor)
        and x._lengths is None,  # ops with "jt" require contiguous JT only
        "jt_all": lambda x: isinstance(
            x, NestedTensor
        ),  # ops with "jt_all" can accept all kinds of JT
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
    if a._size[a._ragged_idx] != b._size[b._ragged_idx]:
        raise RuntimeError(
            f"NestedTensor {func.__name__}: expected {a_name} and {b_name} to have the "
            "same exact offsets tensor."
        )


# returns True if the raggedness-relevant portions of the NT shape
# match those of the specified size
def raggedness_matches(nt, size):
    end = nt._ragged_idx + 1
    return list(nt._size[:end]) == list(size[:end])


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
        return func

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
            check_schema("lhs: any, rhs: any", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)

    return None


def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
        "ragged_size": arg._size[arg._ragged_idx],
    }
    return kwargs


def jagged_unary_pointwise(func, *args, **kwargs):
    return NestedTensor(
        func(args[0]._values, *args[1:], **kwargs), **extract_kwargs(args[0])
    )


def jagged_binary_pointwise(func, *args, **kwargs):
    a, b = args[0], args[1]
    assert isinstance(a, NestedTensor) or isinstance(b, NestedTensor)

    mismatch_error_msg = (
        f"cannot call binary pointwise function {func.__name__} with inputs of shapes "
        f"{a.shape} and {b.shape}"
    )

    # a is NT, b is NT
    if isinstance(a, NestedTensor) and isinstance(b, NestedTensor):
        # ex: (B, j0, D) + (B, j0, D)
        # ex: (B, j0, D) + (B, j0, 1)
        if raggedness_matches(a, b.shape):
            return NestedTensor(
                func(a._values, b._values, *args[2:], **kwargs), **extract_kwargs(a)
            )
        raise RuntimeError(mismatch_error_msg)

    # either a is NT or b is NT at this point
    a_is_nt = isinstance(a, NestedTensor)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)

    # === Handle broadcasting across the batch / ragged dims ===

    # Easy case: take advantage of pre-existing broadcasting logic
    # when NT dim > non-NT dim
    # ex: (B, j0, D_0, D_1) + (D_0, D_1) -> (B, j0, D_0, D_1)
    # ex: (B, j0, D_0, D_1) + (1, D_0, D_1) -> (B, j0, D_0, D_1)
    # ex: (B, j0, 1, 1) + (D_0, D_1) -> (B, j0, D_0, D_1)
    # ex: (B, j0, 1, 1) + (1, D_0, D_1) -> (B, j0, D_0, D_1)
    if (a_is_nt and a.dim() > b.dim()) or (not a_is_nt and b.dim() > a.dim()):
        arg1 = a._values if a_is_nt else a
        arg2 = b._values if not a_is_nt else b
        return NestedTensor(func(arg1, arg2, *args[2:], **kwargs), **extracted_kwargs)

    # Harder case: do manual broadcasting over unbound components
    # when NT dim == non-NT dim
    # ex: (B, j0, D_0, D_1) + (B, 1, D_0, D_1) -> (B, j0, D_0, D_1)
    if a.dim() == b.dim():
        # ex: (B, j0, D_0, D_1) + (1, 1, D_0, D_1) -> should
        # be (B, j0, D_0, D_1) but not yet supported
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(mismatch_error_msg)

        # need to use offsets to broadcast across ragged dim properly
        # NB: inefficient fallback here; Triton codegen can help this
        # TODO: Make this work with autograd
        outputs = []
        for a_comp, b_comp in zip(a.unbind(), b.unbind()):
            outputs.append(func(a_comp, b_comp, *args[2:], **kwargs))
        new_values = torch.cat(outputs, dim=0)
        return NestedTensor(new_values, **extracted_kwargs)

    # ex: (B, j0, D_0, D_1) + (A, B, 1, D_0, D_1) -> error because this breaks the invariant
    # that ragged dim is wrt left-most batch dim
    raise RuntimeError(mismatch_error_msg)


def jagged_torch_function(func, *args, **kwargs):
    # Handle SDPA specially since it's CompositeImplicit. We don't want
    # the nestedness of the inputs to affect the kernel choice, so unwrap
    # the NTs here before passing to SDPA -> rewrap the output as NT.
    if func is torch._C._nn.scaled_dot_product_attention:
        t_args = [t._values if isinstance(t, NestedTensor) else t for t in args]
        t_kwargs = {
            k: v._values if isinstance(v, NestedTensor) else v
            for k, v in kwargs.items()
        }

        output = func(*t_args, **t_kwargs)
        return NestedTensor(output, **extract_kwargs(args[0]))

    # Handle flatten() here because it's CompositeImplicit.
    if func.__name__ == "flatten":

        def _flatten_sig(input, start_dim=0, end_dim=-1):
            pass

        _, new_kwargs = normalize_function(
            _flatten_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        inp = new_kwargs.pop("input")
        new_kwargs["start_dim"] = _wrap_jagged_dim(
            inp.dim(), new_kwargs["start_dim"], "flatten"
        )
        new_kwargs["end_dim"] = _wrap_jagged_dim(
            inp.dim(), new_kwargs["end_dim"], "flatten"
        )

        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))

    raise NotImplementedError(func)


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
        return len(args[0]._size)

    if func == torch.ops.aten.sym_numel.default:
        return args[0]._values.numel()

    if func == torch.ops.aten.sym_stride.default:
        return args[0]._strides

    if func == torch.ops.aten.sym_storage_offset.default:
        return 0


@register_jagged_func(torch.ops.prim.layout.default, "self: jt")
def prim_layout_default(func, *args, **kwargs):
    return torch.jagged


@register_jagged_func(
    [torch.ops.aten.size.default],
    "self: jt",
)
def tensor_attr_unsupported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.size.default:
        raise RuntimeError(
            "NestedTensors does not support directly calling torch.ops.aten.size "
            "please use `nested_tensor.size()` instead."
        )


@register_jagged_func(torch.ops.aten.is_contiguous.default, "self: jt_all")
def is_contiguous_general(func, *args, **kwargs):
    from torch._prims_common import is_contiguous_for_memory_format

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    # If created from narrow() check if offsets and lengths make it possibly contiguous
    if inp.lengths() is not None:
        orig_dim = inp.values().shape[0] / inp.lengths().shape[0]
        if torch.any(inp.lengths()[1:-1].ne(orig_dim)):
            return False
        if torch.any(
            inp.offsets()[1:]
            - torch.arange(
                1, inp.lengths().shape[0], device=inp.device, dtype=torch.int64
            )
            * orig_dim
        ):
            return False
        if inp.offsets()[0] + inp.lengths()[0] != orig_dim:
            return False

    new_kwargs["memory_format"] = new_kwargs.get(
        "memory_format", torch.contiguous_format
    )
    if new_kwargs["memory_format"] == torch.preserve_format:
        return True
    return is_contiguous_for_memory_format(inp.values(), **new_kwargs)


register_jagged_func(
    torch.ops.aten.is_contiguous.memory_format, "self: jt_all, memory_format: any?"
)(is_contiguous_general)


@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    weight = new_kwargs["weight"]
    bias = new_kwargs["bias"]

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.linear_backward.default,
    "self: jt, grad_output: jt, weight: t, output_mask: any",
)
def linear_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")
    weight = new_kwargs.pop("weight")

    check_ragged_dim_same(func, inp, "self", grad_output, "grad_output")
    ds = NestedTensor(
        torch.mm(grad_output._values, weight), **extract_kwargs(grad_output)
    )
    dw = torch.mm(grad_output._values.T, inp._values)
    db = None  # NYI: gradient for bias, need to reduce over ragged dim
    return (ds, dw, db)


@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt")
def to_copy_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # don't change layout
    new_kwargs.pop("layout")

    new_values = func(inp._values, **new_kwargs)
    # NB: Purposefully keep offsets on the old device.
    return NestedTensor(new_values, **extract_kwargs(inp))


register_jagged_func(
    [
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.randn_like.default,
        torch.ops.aten.detach.default,
    ],
    "self: jt",
)(jagged_unary_pointwise)


register_jagged_func(
    torch.ops.aten._softmax.default, "self: jt, dim: any, half_to_float: any"
)(jagged_unary_pointwise)


@register_jagged_func(
    torch.ops.aten.native_dropout.default, "self: jt, float: any, train: any?"
)
def native_dropout_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    out1, out2 = func(inp._values, **new_kwargs)
    return (
        NestedTensor(out1, **extract_kwargs(inp)),
        NestedTensor(out2, **extract_kwargs(inp)),
    )


@register_jagged_func(
    torch.ops.aten.native_dropout_backward.default,
    "grad_output: jt, mask: jt, scale: any",
)
def native_dropout_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_output = new_kwargs.pop("grad_output")
    mask = new_kwargs.pop("mask")
    return NestedTensor(
        func(grad_output._values, mask._values, **new_kwargs),
        **extract_kwargs(grad_output),
    )


@register_jagged_func(torch.ops.aten.prod.dim_int, "self: jt, dim: any, keepdim: any?")
def prod_dim_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # TODO: Figure out how to handle this better
    # keep_dim is required to keep it in jagged format
    if not new_kwargs["keepdim"]:
        raise RuntimeError("prod(): keepdim=True must be set for NestedTensor")
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp.shape), dim, "prod")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(args[0]))


@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any"
)
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    # hack to split on the last dim
    dim = new_kwargs["dim"]
    if dim != -1:
        raise RuntimeError(
            "split_with_sizes(): only supported for NestedTensor on dim = -1 for now"
        )

    split_sizes = new_kwargs["split_sizes"]
    return [
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in torch.split(values, split_sizes, -1)
    ]


@register_jagged_func(torch.ops.aten.unbind.int, "self: jt_all, dim: any?")
def unbind_int(func, *args, **kwargs):
    # Note that this specializes on the length of the offsets
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dim = new_kwargs["dim"]
    if dim != 0:
        raise RuntimeError("unbind(): only supported for NestedTensor on dim=0")

    inp = new_kwargs.pop("input")
    values = inp._values
    offsets = inp.offsets()
    lengths = inp.lengths()

    if lengths is None:
        split_offsets = torch.cat(
            (
                offsets,
                torch.tensor(
                    [values.shape[0]], device=offsets.device, dtype=offsets.dtype
                ),
            )
        )
        return torch.split(values, split_offsets.diff().tolist())
    return [
        values[offsets[i] : (offsets[i] + lengths[i])] for i in range(lengths.shape[0])
    ]


@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values
    offsets = inp.offsets

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp.shape) + 1, dim, "unsqueeze")
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


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
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(first.shape), dim, "cat")

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(torch.ops.aten.matmul.default, "self: jt, other: any")
def matmul_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    if inp.is_nested and not other.is_nested:
        return NestedTensor(
            func(inp._values, other, **new_kwargs), **extract_kwargs(inp)
        )
    elif inp.is_nested and other.is_nested:
        # BMM with equivalent ragged dims between the two inputs
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other.shape):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))

    raise RuntimeError(
        f"matmul(): not supported between inputs of shapes {inp.shape} and {other.shape}"
    )


@register_jagged_func(
    torch.ops.aten.expand.default, "self: jt, size: any, implicit: any?"
)
def expand_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs["size"]

    assert ("implicit" not in new_kwargs) or (not new_kwargs.pop("implicit"))
    if not raggedness_matches(inp, size):
        raise RuntimeError(f"expand(): cannot expand shape {inp.shape} -> {size}")

    expand_arg = [-1, *size[2:]]
    return NestedTensor(func(inp._values, expand_arg), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.expand_as.default, "self: t, other: jt")
def expand_as_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    return NestedTensor(func(inp, other._values), **extract_kwargs(other))


@register_jagged_func(torch.ops.aten.where.self, "condition: jt, self: jt, other: jt")
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    condition = new_kwargs.pop("condition")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    assert condition.shape == other.shape == inp.shape

    return NestedTensor(
        func(condition._values, inp._values, other._values, **new_kwargs),
        **extract_kwargs(condition),
    )


@register_jagged_func(torch.ops.aten._pin_memory.default, "self: jt, device: any?")
def _pin_memory_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.is_pinned.default, "self: jt, device: any?")
def is_pinned_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(torch.ops.aten.is_same_size.default, "self: jt, other: jt")
def is_same_size_default(func, *args, **kwargs):
    return args[0]._size == args[1]._size


@register_jagged_func(torch.ops.aten.transpose.int, "self: jt, dim0: any, dim1: any")
def transpose_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim0"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim0"], "transpose")
    new_kwargs["dim1"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim1"], "transpose")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.view.default, "self: jt, size: any")
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    # Ensure specified size still includes batch and ragged dims
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f"view(): cannot view shape {inp.shape} as {size}")

    jagged_size = [inp._values.shape[0]] + size[2:]
    return NestedTensor(func(inp._values, jagged_size), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.native_layer_norm.default,
    "input: jt, normalized_shape: any, weight: any?, bias: any?, eps: any",
)
def native_layer_norm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    normalized_shape = new_kwargs["normalized_shape"]

    # Ensure we're not trying to normalize over the ragged dim
    if inp.dim() < 3 or (inp.dim() - len(normalized_shape)) < 2:
        raise RuntimeError(
            "layer_norm(): normalizing over ragged dim not supported for nested tensors"
        )

    output, mean, std = func(inp._values, **new_kwargs)
    return (NestedTensor(output, **extract_kwargs(inp)), mean, std)


@register_jagged_func(torch.ops.aten.select.int, "self: jt, dim: any, index: any")
def select_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "select")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.slice.Tensor,
    "self: jt, dim: any?, start: any?, end: any?, step: any?",
)
def slice_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "slice")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.convolution.default,
    "input: jt, weight: t, bias: t?, stride: any, padding: any, "
    "dilation: any, transposed: any, output_padding: any, groups: any",
)
def convolution_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.mean.dim, "self: jt, dim: any?, keepdim: any, dtype: any?"
)
def mean_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # NB: mean expects dim as a single item list of ints for some reason
    new_kwargs["dim"] = [_wrap_jagged_dim(inp.dim(), new_kwargs["dim"][0], "mean")]

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.stack.default, "tensors: any, dim: any")
def stack_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    tensors = new_kwargs.pop("tensors")
    for t in tensors:
        if not isinstance(t, NestedTensor):
            raise RuntimeError("stack(): expected all nested tensors inputs")

        if t.dim() != tensors[0].dim():
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same dim"
            )

        if not raggedness_matches(t, tensors[0].shape):
            raise RuntimeError(
                "stack(): expected all nested tensors to have the same nested structure"
            )

    new_kwargs["dim"] = _wrap_jagged_dim(
        tensors[0].dim() + 1, new_kwargs["dim"], "stack"
    )

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(
    torch.ops.aten.embedding.default,
    "weight: t, indices: jt, padding_idx: any?, scale_grad_by_freq: any?, sparse: any?",
)
def embedding_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # guaranteed this is non-empty if we got here
    indices = new_kwargs.pop("indices")
    weight = new_kwargs.pop("weight")

    return NestedTensor(
        func(weight, indices._values, **new_kwargs), **extract_kwargs(indices)
    )

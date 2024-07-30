# mypy: allow-untyped-defs
import functools
import math
import operator

import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention

from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
import torch.nn.functional as F
from torch.fx.operator_schemas import normalize_function

__all__: List[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}


# Simplifying assumption: we assume that the batch dim is always the left-most
# dim, and the ragged dim is always the second dim.
def _outer_to_inner_dim(ndim, dim):
    assert dim >= 0 and dim < ndim
    return 0 if dim < 2 else dim - 1


def _wrap_jagged_dim(
    ndim, dim, op_name, convert_to_inner_dim=True, allow_batch_dim=False
):
    from torch._prims_common import canonicalize_dims

    wrapped = canonicalize_dims(ndim, dim)
    if wrapped == 1:
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on dim=1")
    elif wrapped == 0 and not allow_batch_dim:
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on dim=0")
    return _outer_to_inner_dim(ndim, wrapped) if convert_to_inner_dim else wrapped


def _wrap_jagged_dims(ndim, dims, op_name, ragged_idx=1):
    """
    For NestedTensor operators,
    wraps dimensions to non-negative values,
    and returns metadata related to reduction dimension(s).
    """
    from torch._prims_common import canonicalize_dims

    assert isinstance(
        dims, (tuple, list)
    ), f"_wrap_jagged_dims(): cannot iterate over dimensions of type {type(dims)}"

    wrapped_dims = [
        canonicalize_dims(ndim, d) for d in dims
    ]  # convert all indices to non-negative values

    operate_on_batch = 0 in wrapped_dims
    operate_on_ragged = ragged_idx in wrapped_dims
    operate_on_non_batch = any(d != 0 and d != ragged_idx for d in wrapped_dims)

    outer_to_inner_dim = tuple(
        _outer_to_inner_dim(ndim, d) for d in wrapped_dims if d != 0
    )

    return outer_to_inner_dim, operate_on_batch, operate_on_ragged, operate_on_non_batch


def check_schema(schema_str: str, func, *args, **kwargs) -> None:
    named_arg_types = schema_str.split(", ")
    num_optional_args = [x.endswith("?") for x in named_arg_types].count(True)
    min_args = len(named_arg_types) - num_optional_args

    # special case: ellipses allows for any number of unchecked args at the end
    if named_arg_types[-1] == "...":
        named_arg_types = named_arg_types[:-1]
    else:
        if not (len(args) >= min_args and len(args) <= len(named_arg_types)):
            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected at least {min_args} "
                f"arguments and at most {len(named_arg_types)} arguments, but got: "
                f"{len(args)} arguments"
            )

    arg_type_check_fns = {
        "t": lambda x: isinstance(x, torch.Tensor) and not isinstance(x, NestedTensor),
        "jt": lambda x: isinstance(x, NestedTensor)
        and x._lengths is None
        and x._ragged_idx == 1,  # ops with "jt" require contiguous JT only
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

        _check_fn = arg_type_check_fns[normalized_arg_type]

        def check_fn(x, is_optional=is_optional):
            if is_optional:
                return x is None or _check_fn(x)
            else:
                return _check_fn(x)

        if not check_fn(args[i]):
            type_to_desc = {
                "t": "tensor",
                "t?": "optional tensor",
                "jt": "contiguous jagged layout NestedTensor",
                "jt_all": "jagged layout NestedTensor",
                "any": "<any type>",
            }

            raise ValueError(
                f"NestedTensor {func.__name__}({schema_str}): expected {name} to be a "
                f"{type_to_desc[arg_type]}"
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
    nt_ragged = nt._size[:end]
    size_ragged = size[:end]
    return len(nt_ragged) == len(size_ragged) and (
        all(ns == s or s == -1 for ns, s in zip(nt_ragged, size_ragged))
    )


def squeeze_leading_ones(t):
    # Note: [ Squeezing leading ones ]
    #
    # Squeeze leading ones from t.
    #
    # We want:
    #   (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    #   (B, j0, ?, ?) + (1, 1, 1, ?, ?) -> (1, B, j0, ?, ?)  (not yet supported)
    #
    # 1) Squeeze extra ones and grab values from NT
    #   (1, 1, ?, ?) -> (?, ?)   and   (sum(*), ?, ?) -> (B, j0, ?, ?)
    # 2) Do dense broadcasting:
    #   (sum(*), ?, ?) + (?, ?) -> (sum(*), ?, ?)
    # 3) Construct nested tensor
    #   (sum(*), ?, ?) -> (B, j0, ?, ?)
    #
    # If unsqueezing on the 0th dim becomes supported, we would unsqueeze
    # at step (4) and we would need to update this function to record how
    # many ones we unsqueezed.
    while t.shape[0] == 1:
        t = t.squeeze(0)
    return t


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
        num_tensor_args = sum(isinstance(x, torch.Tensor) for x in args)
        if num_tensor_args == 1:
            check_schema("self: jt_all, ...", func, *args, **kwargs)
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema("lhs: any, rhs: any, ...", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)

    return None


def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
        "_metadata_cache": arg._metadata_cache,
        "_ragged_idx": arg._ragged_idx,
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
        "cannot call binary pointwise function {} with inputs of shapes {} and {}"
    )
    # a is NT, b is NT
    if isinstance(a, NestedTensor) and isinstance(b, NestedTensor):
        # ex: (B, j0, D) + (B, j0, D)
        # ex: (B, j0, D) + (B, j0, 1)
        if raggedness_matches(a, b._size):
            return NestedTensor(
                func(a._values, b._values, *args[2:], **kwargs), **extract_kwargs(a)
            )
        raise RuntimeError(mismatch_error_msg.format(func.__name__, a._size, b._size))
    # either a is NT or b is NT at this point
    a_is_nt = isinstance(a, NestedTensor)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)

    # === Handle broadcasting across the batch / ragged dims ===

    # Easy case: take advantage of pre-existing broadcasting logic
    # ex: (B, j0, ?, ?) + (?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (?, ?) -> (B, j0, ?, ?)
    # ex: (B, j0, ?, ?) + (1, 1, ?, ?) -> (B, j0, ?, ?)
    nt, t = (a, b) if a_is_nt else (b, a)
    # See Note: [ Squeezing leading ones ]
    if t.dim() > nt.dim():
        raise NotImplementedError("NYI: broadcasting NT with T with larger dim")
    t_squeezed = squeeze_leading_ones(t)
    if nt.dim() >= t_squeezed.dim() + 2:
        lhs, rhs = (nt._values, t_squeezed) if a_is_nt else (t_squeezed, nt._values)
        return NestedTensor(func(lhs, rhs, *args[2:], **kwargs), **extracted_kwargs)

    # Harder case: do manual broadcasting over unbound components
    # when NT dim == non-NT dim
    # ex: (B, j0, D_0, D_1) + (B, 1, D_0, D_1) -> (B, j0, D_0, D_1)
    if a.dim() == b.dim():
        # ex: (B, j0, D_0, D_1) + (1, 1, D_0, D_1) -> should
        # be (B, j0, D_0, D_1) but not yet supported
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(
                mismatch_error_msg.format(func.__name__, a.shape, b.shape)
            )

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
    raise RuntimeError(mismatch_error_msg.format(func.__name__, a.shape, b.shape))


def jagged_torch_function(func, *args, **kwargs):
    # SDPA has special kernels that handle nested tensors.
    # Dispatch to the correct implementation here
    if func is torch._C._nn.scaled_dot_product_attention:
        return jagged_scaled_dot_product_attention(*args, **kwargs)

    if func.__name__ == "apply_":
        func(args[0]._values, *args[1:], **kwargs)
        return args[0]

    # Handle flatten() here because it's CompositeImplicit.
    if func.__name__ == "flatten":

        def _flatten_sig(input, start_dim=0, end_dim=-1):
            pass

        _, new_kwargs = normalize_function(
            _flatten_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        inp = new_kwargs.pop("input")

        # NB: stay in outer dim space because we're going to redispatch on a NT input
        start_dim = _wrap_jagged_dim(
            inp.dim(), new_kwargs["start_dim"], "flatten", convert_to_inner_dim=False
        )
        end_dim = _wrap_jagged_dim(
            inp.dim(), new_kwargs["end_dim"], "flatten", convert_to_inner_dim=False
        )

        if start_dim == end_dim:
            return inp

        product = functools.reduce(operator.mul, inp.shape[start_dim : end_dim + 1])
        new_shape = (*inp.shape[:start_dim], product, *inp.shape[end_dim + 1 :])

        return inp.reshape(*new_shape)

    raise NotImplementedError(func)


@register_jagged_func(
    [
        torch.ops.aten.is_non_overlapping_and_dense.default,
        torch.ops.aten.sym_size.default,
        torch.ops.aten.dim.default,
        torch.ops.aten.numel.default,
        torch.ops.aten.sym_numel.default,
        torch.ops.aten.sym_stride.default,
        torch.ops.aten.sym_storage_offset.default,
    ],
    "self: jt_all",
)
def tensor_attr_supported_getter(func, *args, **kwargs):
    if func == torch.ops.aten.is_non_overlapping_and_dense.default:
        return False

    if func == torch.ops.aten.sym_size.default:
        return args[0]._size

    if func == torch.ops.aten.dim.default:
        return len(args[0]._size)

    if func in (torch.ops.aten.sym_numel.default, torch.ops.aten.numel.default):
        if args[0]._lengths is not None:
            return int(sum(args[0]._lengths) * math.prod(args[0]._size[2:]))
        return args[0]._values.numel()

    if func == torch.ops.aten.sym_stride.default:
        return args[0]._strides

    if func == torch.ops.aten.sym_storage_offset.default:
        return args[0]._values.storage_offset()


@register_jagged_func(torch.ops.prim.layout.default, "self: jt_all")
def prim_layout_default(func, *args, **kwargs):
    return torch.jagged


@register_jagged_func(
    [torch.ops.aten.size.default],
    "self: jt_all",
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

    # If created from narrow() check for lengths
    if inp.lengths() is not None:
        return False

    new_kwargs["memory_format"] = new_kwargs.get(
        "memory_format", torch.contiguous_format
    )
    if new_kwargs["memory_format"] == torch.preserve_format:
        return True
    return is_contiguous_for_memory_format(inp._values, **new_kwargs)


register_jagged_func(
    torch.ops.aten.is_contiguous.memory_format, "self: jt_all, memory_format: any?"
)(is_contiguous_general)


@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

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
        torch.matmul(grad_output._values, weight), **extract_kwargs(grad_output)
    )
    dw = torch.matmul(grad_output._values.transpose(-2, -1), inp._values)
    db = None  # NYI: gradient for bias, need to reduce over ragged dim
    return (ds, dw, db)


@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt_all")
def to_copy_default(func, *args, **kwargs):
    from .nested_tensor import _tensor_symint_registry

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # don't change layout
    new_kwargs.pop("layout")

    new_values = func(inp._values, **new_kwargs)
    new_offsets = inp._offsets.to(device=new_values.device)
    _tensor_symint_registry[new_offsets] = _tensor_symint_registry[inp._offsets]
    inp_kwargs = extract_kwargs(inp)
    inp_kwargs["offsets"] = new_offsets

    return NestedTensor(new_values, **inp_kwargs)


register_jagged_func(torch.ops.aten.detach.default, "self: jt_all")(
    jagged_unary_pointwise
)


@register_jagged_func(
    [
        torch.ops.aten.empty_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.randn_like.default,
    ],
    "self: jt_all",
)
def like_factory_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # Default layout is technically torch.strided but only jagged is supported here.
    # Rather than force users to specify the layout, assume jagged.
    # This should be set to strided for redispatching on values.
    new_kwargs["layout"] = torch.strided

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.zero_.default, "self: jt_all")
def zero__default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    func(inp._values)
    return inp


@register_jagged_func(
    torch.ops.aten._softmax.default, "self: jt_all, dim: any, half_to_float: any"
)
def _softmax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    if isinstance(new_kwargs["dim"], tuple):
        raise RuntimeError(
            "softmax(): not supported for dimensions of type 'tuple' for NestedTensor"
        )

    inp = new_kwargs.pop("input")

    (
        new_kwargs["dim"],
        reduce_on_batch,
        reduce_on_ragged,
        reduce_on_non_batch,
    ) = _wrap_jagged_dims(
        inp.dim(),
        (new_kwargs["dim"],),
        "softmax",
        inp._ragged_idx,
    )

    if reduce_on_batch:
        raise RuntimeError(
            "softmax(): not supported when reducing across the batch dimension for NestedTensor"
        )

    if reduce_on_ragged and inp._ragged_idx > 1:
        raise RuntimeError(
            "softmax(): not supported when reducing along the ragged dimension for ragged_idx > 1 for NestedTensor"
        )

    if reduce_on_ragged and inp._lengths is not None:
        raise RuntimeError(
            "softmax(): not supported where lengths is not None "
            + "if reducing across the ragged dimension for NestedTensor"
        )

    new_kwargs["dim"] = new_kwargs["dim"][
        0
    ]  # torch.softmax takes in the reduction dimension as an integer

    if reduce_on_ragged:
        padded_softmax_values = torch.nn.functional.softmax(
            torch.ops.aten._jagged_to_padded_dense_forward(
                inp._values.flatten(
                    start_dim=inp._ragged_idx
                ),  # values are required to be 2D tensors for j2pd
                [inp._offsets],
                max_lengths=[inp._max_seqlen],  # max length of ragged dimension
                padding_value=float("-inf"),  # e^-inf = 0
            ),
            dim=inp._ragged_idx,
        )

        softmax_values = torch.ops.aten._padded_dense_to_jagged_forward(
            padded_softmax_values,
            [inp._offsets],
            total_L=inp._values.shape[
                0
            ],  # providing this parameter helps avoid a GPU/CPU sync
        ).reshape(
            -1, *inp._values.shape[1:]
        )  # expand softmax_values back to original shape (inp._values.shape)

        return NestedTensor(softmax_values, **extract_kwargs(inp))

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten._softmax_backward_data.default,
    "grad_output: jt, output: jt, dim: any, input_dtype: any",
)
def _softmax_backward(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_out = new_kwargs.pop("grad_output")
    output = new_kwargs.pop("output")
    return NestedTensor(
        func(grad_out._values, output._values, **new_kwargs), **extract_kwargs(grad_out)
    )


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
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size), dim, "prod")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(args[0]))


@register_jagged_func(
    torch.ops.aten.split.Tensor, "self: jt, split_size: any, dim: any"
)
def split_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "split")

    return tuple(
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    )


@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any"
)
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "split_with_sizes"
    )

    return [
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    ]


@register_jagged_func(
    torch.ops.aten.narrow.default, "self: jt, dim: any, start: any, length: any"
)
def narrow(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    dim = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], "narrow")
    values = func(
        inp._values,
        dim=dim,
        start=new_kwargs["start"],
        length=new_kwargs["length"],
    )
    return NestedTensor(values, **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.chunk.default, "self: jt, chunks: any, dim: any?")
def chunk_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "chunk", allow_batch_dim=True
    )

    if new_kwargs["dim"] == 0:
        chunks = new_kwargs["chunks"]
        dim0_size = inp._size[0]
        chunk_size = math.ceil(dim0_size / chunks)

        # get _offsets of the chunks
        lengths = inp._offsets.diff()
        chunked_lengths = lengths.chunk(chunks)
        chunked_offsets = [torch.cumsum(x, dim=0) for x in chunked_lengths]
        chunked_offsets = [F.pad(x, (1, 0), value=0) for x in chunked_offsets]
        nested_kwargs = [
            {"offsets": per_offsets, "_ragged_idx": inp._ragged_idx}
            for per_offsets in chunked_offsets
        ]

        # get _values of the chunks
        split_sizes = [x.sum().item() for x in chunked_lengths]
        chunk_values = inp._values.split(split_sizes)

        return [
            NestedTensor(values=chunk_values[i], **(nested_kwargs[i]))
            for i in range(0, chunk_size)
        ]
    else:
        return [
            NestedTensor(values=x, **extract_kwargs(inp))
            for x in func(inp._values, **new_kwargs)
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
    values = inp.values()
    offsets = inp.offsets()
    lengths = inp.lengths()
    ragged_idx = inp._ragged_idx

    if lengths is None:
        return torch.split(values, offsets.diff().tolist(), dim=(ragged_idx - 1))

    if ragged_idx <= 0:
        raise RuntimeError(
            "unbind(): nested tensor ragged_idx out of bounds (should be >= 1)"
        )
    for i in range(lengths.shape[0]):
        if offsets[i] + lengths[i] > values.shape[ragged_idx - 1]:
            raise RuntimeError(
                "unbind(): nested tensor offsets and lengths do not match ragged_idx dimension"
            )
    return [
        torch.narrow(values, dim=(ragged_idx - 1), start=offsets[i], length=lengths[i])
        for i in range(lengths.shape[0])
    ]


@register_jagged_func(torch.ops.aten.squeeze.dim, "self: jt, dim: any")
def squeeze_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size), new_kwargs["dim"], "squeeze")
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size) + 1, dim, "unsqueeze")
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
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other._size):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))

    raise RuntimeError(
        f"matmul(): not supported between inputs of shapes {inp._size} and {other.shape}"
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
        raise RuntimeError(f"expand(): cannot expand shape {inp._size} -> {size}")

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

    assert condition._size == other._size == inp._size

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


@register_jagged_func(
    torch.ops.aten.is_same_size.default, "self: jt_all, other: jt_all"
)
def is_same_size_default(func, *args, **kwargs):
    return args[0]._size == args[1]._size


@register_jagged_func(
    torch.ops.aten.sum.dim_IntList,
    "self: jt_all, dim: any?, keepdim: any?, dtype: any?",
)
def sum_dim_IntList(func, *args, **kwargs):
    """
    Performs a sum along the provided tensor dimension.
    Returns a dense tensor if the ragged dimension is reduced away, else returns a nested tensor.
    """
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    (
        new_kwargs["dim"],
        reduce_on_batch,
        reduce_on_ragged,
        reduce_on_non_batch,  # noqa: UFMT
    ) = _wrap_jagged_dims(
        inp.dim(),
        new_kwargs["dim"],
        "sum",
        inp._ragged_idx,
    )

    if reduce_on_ragged and inp._lengths is not None:
        raise RuntimeError(
            "sum(): not supported where lengths is not None "
            + "if reducing across the ragged dimension for NestedTensor"
        )

    if reduce_on_ragged:  # raggedness reduced away --> return dense tensor
        if (
            reduce_on_batch
        ):  # reduction cases: (batch, ragged), (batch, ragged, non-batch), etc.
            out = func(
                inp._values, **new_kwargs
            )  # no need to read offsets --> apply sum directly on values
        else:
            if (
                reduce_on_non_batch
            ):  # invalid reduction cases: (ragged, non-batch), etc.
                raise RuntimeError(
                    "sum(): not supported along a ragged and non-batch dimension for NestedTensor"
                )
            # reduction cases: (ragged)
            values_ragged_dim_outer = inp._values.permute(
                inp._ragged_idx - 1,  # outer dimension
                *range(0, inp._ragged_idx - 1),
                *range(inp._ragged_idx, inp.dim() - 1),
            )  # shift reduction dimension of values backward to outer dimension

            # _jagged_to_padded_dense_forward requires values to be a 2D tensor
            # with the ragged dimension as the 0th dimension
            padded = torch.ops.aten._jagged_to_padded_dense_forward(
                values_ragged_dim_outer.reshape(values_ragged_dim_outer.shape[0], -1),
                [inp._offsets],
                max_lengths=[inp._max_seqlen],
            )

            padded_ragged_dim_original = padded.view(
                padded.shape[0],
                inp._max_seqlen,
                *values_ragged_dim_outer.shape[
                    1:
                ],  # expand non-batch dimensions of padded tensor
            ).permute(
                0,
                *range(2, inp._ragged_idx + 1),
                1,
                *range(inp._ragged_idx + 1, inp.dim()),
            )  # shift reduction dimension of padded tensor forward to original ragged dimension

            out = torch.sum(
                padded_ragged_dim_original,
                dim=inp._ragged_idx,
            )  # need to read offsets --> pad jagged dimension and apply sum

        if new_kwargs["keepdim"]:
            out = out.unsqueeze(0)
        return out
    else:  # raggedness preserved --> return nested tensor
        if (
            reduce_on_batch
        ):  # invalid reduction cases: (batch), (batch, non-batch), etc.
            raise RuntimeError(
                "sum(): not supported along the batch dimension but not the ragged dimension for NestedTensor"
            )
        # reduction cases: (non-batch), (non-batch, non-batch), etc.
        return NestedTensor(
            func(inp._values, **new_kwargs), **extract_kwargs(inp)
        )  # apply sum directly on values


@register_jagged_func(
    torch.ops.aten.transpose.int, "self: jt_all, dim0: any, dim1: any"
)
def transpose_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    from torch._prims_common import canonicalize_dims

    inp = new_kwargs.pop("input")
    dim0, dim1 = canonicalize_dims(inp.dim(), (new_kwargs["dim0"], new_kwargs["dim1"]))

    if inp._lengths is not None:
        raise ValueError(
            "transpose(): not supported on jagged layout nested tensor with holes"
        )

    # To support the SDPA API, inputs need to have the ragged idx transposed to dim 2
    # instead of 1, although the internal Flash and mem-effn implementations will
    # use the inputs with raggedness in dim 1.
    if dim0 == inp._ragged_idx or dim1 == inp._ragged_idx:
        if dim0 == 0 or dim1 == 0:
            raise ValueError(
                "Transpose is not supported on the batch dimension for jagged NT"
            )
        if dim0 == inp._ragged_idx:
            to_dim = dim1
        else:
            to_dim = dim0
        inp_kwargs = extract_kwargs(inp)
        inp_kwargs["_ragged_idx"] = to_dim
        return NestedTensor(
            inp.values().transpose(
                _outer_to_inner_dim(len(inp._size), dim0),
                _outer_to_inner_dim(len(inp._size), dim1),
            ),
            **inp_kwargs,
        )

    new_kwargs["dim0"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim0"], "transpose")
    new_kwargs["dim1"] = _wrap_jagged_dim(inp.dim(), new_kwargs["dim1"], "transpose")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default],
    "self: jt_all, size: any",
)
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    if inp._ragged_idx != 1 and tuple(inp._size) != tuple(size):
        raise RuntimeError(
            f"view(): does not support ragged_idx != 1 except when inp._size == size. "
            f"inp._size is ({inp._size}) and size is ({size})."
        )

    # Ensure specified size still includes batch and ragged dims
    if len(size) < 3 or not raggedness_matches(inp, size):
        raise RuntimeError(f"view(): cannot view shape {inp._size} as {size}")

    # outer size: the size of the NT, e.g. [3, j0, 10]
    # inner size: the size of the values, e.g. [8, 10] (e.g. for offsets = [0, 3, 5, 8])
    # this function gets inner_size[inner_idx] for a given inner_idx.
    #
    # example: for outer size [a, b, c, j0, d, e, f]
    #                         assume that j0 is ragged, other are concrete integers
    #                         and ragged_idx=3
    # inner size will be      [b, c, inp._values.size(ragged_idx), d, e, f]
    # therefore:
    #    inner_size[0] = outer_size[1]
    #    inner_size[1] = outer_size[2]
    #    inner_size[0] = inp._values.size(ragged_idx - 1)
    #    inner_size[3] = outer_size[4]
    #    inner_size[4] = outer_size[5]
    def get_inner_size(inner_idx):
        nonlocal inp, size
        if inner_idx == inp._ragged_idx - 1:
            return inp._values.size(inner_idx)
        else:
            return size[inner_idx + 1]

    inner_size = [get_inner_size(i) for i in range(len(size) - 1)]

    return NestedTensor(func(inp._values, inner_size), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.native_layer_norm.default,
    "input: jt_all, normalized_shape: any, weight: any?, bias: any?, eps: any",
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


@register_jagged_func(
    torch.ops.aten.native_layer_norm_backward.default,
    "grad_out: jt, input: jt, normalized_shape: any, mean: any, rstd: any, weight: any?, bias: any?, output_mask: any",
)
def native_layer_norm_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_out = new_kwargs.pop("grad_out")
    inp = new_kwargs.pop("input")
    d_input, d_gamma, d_beta = func(grad_out._values, inp._values, **new_kwargs)
    if d_input is None:
        return (None, d_gamma, d_beta)

    return (NestedTensor(d_input, **extract_kwargs(inp)), d_gamma, d_beta)


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
    torch.ops.aten.mean.dim, "self: jt_all, dim: any?, keepdim: any?, dtype: any?"
)
def mean_dim(func, *args, **kwargs):
    """
    Performs a mean along the provided tensor dimension.
    Returns a dense tensor if the ragged dimension is reduced away, else returns a nested tensor.
    """
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    if len(new_kwargs["dim"]) > 1:
        raise RuntimeError(
            "mean(): not supported across multiple dimensions for NestedTensor"
        )

    inp = new_kwargs.pop("input")

    (
        new_kwargs["dim"],
        reduce_on_batch,
        reduce_on_ragged,
        reduce_on_non_batch,  # noqa: UFMT
    ) = _wrap_jagged_dims(
        inp.dim(),
        new_kwargs["dim"],
        "mean",
        inp._ragged_idx,
    )

    if reduce_on_batch:
        raise RuntimeError(
            "mean(): not supported along the batch dimension but not the ragged dimension for NestedTensor"
        )

    if reduce_on_ragged and inp._lengths is not None:
        raise RuntimeError(
            "mean(): not supported where lengths is not None "
            + "if reducing across the ragged dimension for NestedTensor"
        )

    if not new_kwargs["keepdim"]:
        raise RuntimeError("mean(): not supported when keepdim=False for NestedTensor")

    if reduce_on_ragged:  # raggedness reduced away
        torch_sum = torch.sum(inp, dim=inp._ragged_idx, keepdim=new_kwargs["keepdim"])

        # for every non-batch dimension,
        #   unsqueeze lengths into the same shape as the PyTorch sum,
        #   as the extra dimensions must all be divided by the same length
        lengths = inp._offsets.diff()
        for _ in range(inp.dim() - 2):
            lengths = lengths.unsqueeze(-1)

        return torch_sum / lengths.broadcast_to(torch_sum.shape)

    return NestedTensor(
        func(inp._values, **new_kwargs), **extract_kwargs(inp)
    )  # raggedness preserved


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


@register_jagged_func(
    [
        torch.ops.aten.values.default,
        torch.ops.aten._nested_get_values.default,
    ],
    "self: jt_all",
)
def values_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # TODO: Handle inference mode properly.
    # See https://github.com/pytorch/pytorch/issues/112024#issuecomment-1779554292
    return inp._values.detach()


@register_jagged_func(
    torch.ops.aten._nested_view_from_jagged.default,
    "values: t, offsets: t, dummy: jt_all, lengths: t?, ragged_idx: any?, min_seqlen: t?, max_seqlen: t?",
)
def _nested_view_from_jagged_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    values, offsets, lengths = (
        new_kwargs["input"],
        new_kwargs["offsets"],
        new_kwargs["lengths"],
    )
    ragged_idx = new_kwargs["ragged_idx"]
    min_seqlen = new_kwargs["min_seqlen"]
    max_seqlen = new_kwargs["max_seqlen"]
    metadata_cache = {}
    if min_seqlen is not None:
        metadata_cache["min_seqlen"] = min_seqlen
    if max_seqlen is not None:
        metadata_cache["max_seqlen"] = max_seqlen

    return NestedTensor(
        values,
        offsets,
        lengths=lengths,
        _ragged_idx=ragged_idx,
        _metadata_cache=metadata_cache,
    )


@register_jagged_func(torch.ops.aten._nested_get_offsets.default, "self: jt_all")
def _nested_get_offsets(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._offsets


@register_jagged_func(torch.ops.aten._nested_get_lengths.default, "self: jt_all")
def _nested_get_lengths(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._lengths


@register_jagged_func(torch.ops.aten._nested_get_ragged_idx.default, "self: jt_all")
def _nested_get_ragged_idx(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._ragged_idx


@register_jagged_func(torch.ops.aten._nested_get_min_seqlen.default, "self: jt_all")
def _nested_get_min_seqlen(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._metadata_cache.get("min_seqlen", None)


@register_jagged_func(torch.ops.aten._nested_get_max_seqlen.default, "self: jt_all")
def _nested_get_max_seqlen(func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._metadata_cache.get("max_seqlen", None)


# Make the dummy available on the C++ side.
@register_jagged_func(torch.ops.aten._nested_get_jagged_dummy.default, "self: any")
def _nested_get_jagged_dummy(func, *args, **kwargs):
    from torch.nested._internal.nested_tensor import _nt_view_dummy

    return _nt_view_dummy()


with torch.library._scoped_library("aten", "IMPL") as aten:
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CPU")
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CUDA")
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "Meta")

# mypy: allow-untyped-defs
import functools
import math
import operator
from typing import *  # noqa: F403

import torch
import torch.nn.functional as F
from torch.fx.operator_schemas import normalize_function
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention

from .nested_tensor import NestedTensor


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
    while t.dim() > 0 and t.shape[0] == 1:
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
        from torch.fx.experimental.symbolic_shapes import is_nested_int

        # No pointwise ops legitimately accept nested int inputs. Without this check,
        # they will be incorrectly interpreted as tensors.
        # See https://github.com/pytorch/pytorch/issues/138496
        for arg in args:
            if is_nested_int(arg):
                raise RuntimeError(
                    f"NestedTensor {func.__name__}: invalid argument {arg}"
                )

        # Assume there aren't additional tensors that aren't the "unary/binary" args
        num_tensor_args = sum(isinstance(x, torch.Tensor) for x in args)
        if num_tensor_args == 1:
            # Build up the check schema string. The first tensor arg is assumed to be
            # an NJT and other args are sent through as-is.
            schema_parts = []
            for arg in func._schema.arguments:
                if isinstance(arg.type, torch.TensorType):
                    schema_parts.append(f"{arg.name}: jt_all")
                    break
                else:
                    schema_parts.append(f"{arg.name}: any")
            schema_parts.append("...")
            check_schema_str = ", ".join(schema_parts)
            check_schema(check_schema_str, func, *args, **kwargs)
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema("lhs: any, rhs: any, ...", func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)

    return None


def extract_kwargs(arg):
    kwargs = {
        "offsets": arg.offsets(),
        "lengths": arg.lengths(),
        "_metadata_cache": arg._metadata_cache,
        "_ragged_idx": arg._ragged_idx,
    }
    return kwargs


def jagged_unary_pointwise(func, *args, **kwargs):
    # assume if we get here that there is a single NJT input in the args
    njt = next(arg for arg in args if isinstance(arg, NestedTensor))
    return NestedTensor(
        func(*(arg._values if arg is njt else arg for arg in args), **kwargs),
        **extract_kwargs(njt),
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

    # Harder case: do manual broadcasting when NT dim == non-NT dim
    # ex: (B, j0, D_0, D_1) + (B, 1, D_0, D_1) -> (B, j0, D_0, D_1)
    if a.dim() == b.dim():
        # ex: (B, j0, D_0, D_1) + (1, 1, D_0, D_1) -> should
        # be (B, j0, D_0, D_1) but not yet supported
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(
                mismatch_error_msg.format(func.__name__, a.shape, b.shape)
            )

        from .nested_tensor import nested_from_padded

        # handle broadcasting via padded dense -> jagged conversion
        min_seqlen = nt._maybe_min_seqlen
        max_seqlen = nt._maybe_max_seqlen
        padded_max_S = max_seqlen
        total_L = nt._values.shape[nt._ragged_idx - 1]
        if padded_max_S is None:
            # use upper bound on max seqlen if it's not present
            padded_max_S = total_L

        # convert dense tensor -> jagged
        t = t.expand(
            [x if i != nt._ragged_idx else padded_max_S for i, x in enumerate(t.shape)]
        )
        t_as_nt = nested_from_padded(
            t,
            offsets=nt._offsets,
            ragged_idx=nt._ragged_idx,
            sum_S=total_L,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        )

        # function call with two NJTs
        lhs, rhs = (nt, t_as_nt) if a_is_nt else (t_as_nt, nt)
        return func(lhs, rhs, *args[2:], **kwargs)

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

        _, new_kwargs = normalize_function(  # type: ignore[misc]
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

    # Handle nested-specific input validation for CompositeImplicit rms_norm
    if func.__name__ == "rms_norm":

        def _rms_norm_sig(input, normalized_shape, weight=None, eps=None):
            pass

        _, new_kwargs = normalize_function(  # type: ignore[misc]
            _rms_norm_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )

        inp = new_kwargs.pop("input")
        normalized_shape = new_kwargs.pop("normalized_shape")

        # can't normalize over the ragged dim (yet)
        max_normalizable = inp.dim() - inp._ragged_idx - 1
        if len(normalized_shape) > max_normalizable:
            raise ValueError(
                "rms_norm(): Normalization over the ragged dim not supported for nested tensors"
            )

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

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
            "NestedTensor does not support directly calling torch.ops.aten.size; "
            "please use `nested_tensor.size()` instead."
        )


@register_jagged_func(torch.ops.aten.is_contiguous.default, "self: jt_all")
def is_contiguous_general(func, *args, **kwargs):
    from torch._prims_common import is_contiguous_for_memory_format

    _, new_kwargs = normalize_function(  # type: ignore[misc]
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


@register_jagged_func(
    torch.ops.aten.clone.default, "input: jt_all, memory_format: any?"
)
def clone_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_meta = extract_kwargs(inp)

    if inp._lengths is not None:
        if new_kwargs["memory_format"] == torch.contiguous_format:
            # need to copy to remove "holes" non-contiguity / lengths metadata
            # TODO: write a kernel for this
            from .nested_tensor import jagged_from_list

            # TODO: We probably want the output to have the same ragged structure / nested int.
            assert (
                inp._ragged_idx == 1
            ), "NJT with ragged_idx != 1 not supported for contiguous clone"
            contig, _ = jagged_from_list(inp.unbind(), offsets=None)
            return contig

    return NestedTensor(func(inp._values, **new_kwargs), **new_meta)


@register_jagged_func(torch.ops.aten.linear.default, "input: jt, weight: t, bias: t?")
def linear_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.linear_backward.default,
    "self: jt, grad_output: jt, weight: t, output_mask: any",
)
def linear_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")
    weight = new_kwargs.pop("weight")
    output_mask = new_kwargs.pop("output_mask")

    ds, dw, db = None, None, None
    check_ragged_dim_same(func, inp, "self", grad_output, "grad_output")
    if output_mask[0]:
        ds = NestedTensor(
            torch.matmul(grad_output._values, weight), **extract_kwargs(grad_output)
        )
    if output_mask[1]:
        dw = torch.matmul(grad_output._values.transpose(-2, -1), inp._values)
    if output_mask[2]:
        db = grad_output._values.sum(0)
    return (ds, dw, db)


@register_jagged_func(torch.ops.aten.to.dtype, "input: jt_all, dtype: any")
def to_dtype(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten._to_copy.default, "self: jt_all")
def to_copy_default(func, *args, **kwargs):
    from .nested_tensor import _tensor_symint_registry

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    # don't change layout
    new_kwargs.pop("layout")

    new_values = func(inp._values, **new_kwargs)
    new_offsets = inp._offsets.to(device=new_values.device)
    new_lengths = None
    if inp._lengths is not None:
        new_lengths = inp._lengths.to(device=new_values.device)

    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import (
        FunctionalTensor,
        mb_unwrap_functional_tensor,
    )

    ragged_source = inp._offsets if inp._lengths is None else inp._lengths
    new_thing = new_offsets if new_lengths is None else new_lengths
    if isinstance(new_thing, (FakeTensor, FunctionalTensor)):
        # Temporary hack until we have the union find
        tgt = mb_unwrap_functional_tensor(new_thing)
        src = mb_unwrap_functional_tensor(ragged_source)
        tgt.nested_int_memo = src.nested_int_memo
    else:
        _tensor_symint_registry[new_thing] = _tensor_symint_registry[ragged_source]
    inp_kwargs = extract_kwargs(inp)
    inp_kwargs["offsets"] = new_offsets
    inp_kwargs["lengths"] = new_lengths

    output = NestedTensor(new_values, **inp_kwargs)
    return output


@register_jagged_func(
    torch.ops.aten.copy_.default, "self: jt_all, src: jt_all, non_blocking: any?"
)
def copy_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    src = new_kwargs.pop("src")
    if inp._size != src._size:
        raise RuntimeError(
            "copy_ only supports Nested Tensors that have same size and the exact same offset tensor."
        )
    inp.values().copy_(src.values())
    return inp


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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    func(inp._values)
    return inp


@register_jagged_func(
    torch.ops.aten._softmax.default, "self: jt_all, dim: any, half_to_float: any"
)
def _softmax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
        _reduce_on_non_batch,
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
                inp._values.reshape(
                    inp._values.shape[0], -1
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
        chunked_offsets = [F.pad(x, (1, 0), value=0) for x in chunked_offsets]  # type: ignore[arg-type]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    new_kwargs["dim"] = _wrap_jagged_dim(len(inp._size), new_kwargs["dim"], "squeeze")
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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


@register_jagged_func(torch.ops.aten.matmul.default, "self: jt_all, other: any")
def matmul_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    def _unbind_impl(a, b):
        return [
            func(a_comp, b_comp) for (a_comp, b_comp) in zip(a.unbind(), b.unbind())
        ]

    def _padded_impl(a, b):
        assert a.is_nested and not b.is_nested
        nt = a

        from .nested_tensor import nested_from_padded

        min_seqlen = nt._maybe_min_seqlen
        max_seqlen = nt._maybe_max_seqlen
        padded_max_S = max_seqlen
        total_L = nt._values.shape[nt._ragged_idx - 1]
        if padded_max_S is None:
            # use upper bound on max seqlen if it's not present
            padded_max_S = total_L

        padded_shape = (
            *nt.shape[: nt._ragged_idx],
            padded_max_S,
            *nt.shape[nt._ragged_idx + 1 :],
        )
        padded_nt = nt.to_padded_tensor(0.0, output_size=padded_shape)
        return nested_from_padded(
            func(padded_nt, b),
            offsets=nt._offsets,
            ragged_idx=nt._ragged_idx,
            sum_S=total_L,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        )

    # TODO: Back these with proper kernels (e.g. grouped GEMM)
    # NJT x dense
    if inp.is_nested and not other.is_nested:
        # (B, j1, D) x (B, D, E) => (B, j1, E)
        if inp.dim() >= 3 and inp.dim() == other.dim():
            # convert to padded for this
            return _padded_impl(inp, other)
        # Support broadcasting the dense:
        # (B, j1, D) x (D, E) => (B, j1, E)
        # (B, j1, D, E) x (E, F) => (B, j1, D, F)
        # etc.
        elif other.dim() == 2 and inp.dim() > other.dim():
            return NestedTensor(
                func(inp._values, other, **new_kwargs), **extract_kwargs(inp)
            )
    # NJT x NJT
    elif inp.is_nested and other.is_nested:
        # Support ragged batch dim:
        # (B, j1, D, E) x (B, j1, E, F) => (B, j1, D, F), etc.
        if inp.dim() > 3 and other.dim() > 3 and raggedness_matches(inp, other._size):
            return NestedTensor(func(inp._values, other._values), **extract_kwargs(inp))
        # Support reducing over ragged with dense output:
        # (B, D, j1) x (B, j1, E) => (B, D, E)
        elif (
            inp.dim() == 3
            and other.dim() == 3
            and inp._ragged_idx == 2
            and other._ragged_idx == 1
            and inp.size(inp._ragged_idx) == other.size(other._ragged_idx)
        ):
            # do unbind for this; can't use padded conversion due to j1 in last dim
            return torch.stack(_unbind_impl(inp, other))

    raise RuntimeError(
        f"matmul(): not supported between inputs of shapes {inp._size} and {other.shape}"
    )


@register_jagged_func(torch.ops.aten.bmm.default, "self: jt_all, mat2: any")
def bmm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("mat2")

    if inp.dim() != 3:
        raise ValueError("bmm(): input must be 3D")
    if other.dim() != 3:
        raise ValueError("bmm(): mat2 must be 3D")

    return matmul_default(torch.ops.aten.matmul.default, inp, other)


@register_jagged_func(
    torch.ops.aten.expand.default, "self: jt, size: any, implicit: any?"
)
def expand_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    return NestedTensor(func(inp, other._values), **extract_kwargs(other))


@register_jagged_func(torch.ops.aten.where.self, "condition: jt, self: jt, other: jt")
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.is_pinned.default, "self: jt, device: any?")
def is_pinned_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.is_same_size.default, "self: jt_all, other: jt_all"
)
def is_same_size_default(func, *args, **kwargs):
    return args[0]._size == args[1]._size


@register_jagged_func(torch.ops.aten.sum.default, "self: jt_all, dtype: any?")
def sum_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.sum.dim_IntList,
    "self: jt_all, dim: any?, keepdim: any?, dtype: any?",
)
def sum_dim_IntList(func, *args, **kwargs):
    """
    Performs a sum along the provided tensor dimension.
    Returns a dense tensor if the ragged dimension is reduced away, else returns a nested tensor.
    """
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")

    (
        new_kwargs["dim"],
        reduce_on_batch,
        reduce_on_ragged,
        reduce_on_non_batch,
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
            max_seqlen = inp._values.shape[inp._ragged_idx - 1]
            if inp._max_seqlen_tensor is not None:
                max_seqlen = inp._max_seqlen

            padded = torch.ops.aten._jagged_to_padded_dense_forward(
                values_ragged_dim_outer.reshape(values_ragged_dim_outer.shape[0], -1),
                [inp._offsets],
                max_lengths=[max_seqlen],
            )

            padded_ragged_dim_original = padded.view(
                padded.shape[0],
                max_seqlen,
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
            out = out.unsqueeze(inp._ragged_idx)
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    from torch._prims_common import canonicalize_dims

    inp = new_kwargs.pop("input")
    dim0, dim1 = canonicalize_dims(inp.dim(), (new_kwargs["dim0"], new_kwargs["dim1"]))

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


@register_jagged_func(torch.ops.aten.permute.default, "self: jt_all, dims: any")
def permute_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    dims = new_kwargs.pop("dims")
    inp_kwargs = extract_kwargs(inp)
    inp_dim = len(inp._size)

    # The first two checks are the same as the checks in the normal permute implementation
    if inp_dim != len(dims):
        raise ValueError(
            f"permute(): number of dimensions in the tensor input ({inp_dim}) "
            + f"does not match the length of the desired ordering of dimensions ({len(dims)}).",
        )

    from torch._prims_common import canonicalize_dims

    canonicalized_dims = canonicalize_dims(inp_dim, dims)

    if len(canonicalized_dims) != len(set(canonicalized_dims)):
        raise ValueError("permute(): duplicate dims are not allowed.")

    if inp._lengths is not None:
        raise ValueError(
            "permute(): not supported on jagged layout nested tensor with holes"
        )
    if canonicalized_dims[0] != 0:
        raise ValueError(
            "Permute is not supported on the batch dimension for jagged NT"
        )
    inp_kwargs["_ragged_idx"] = canonicalized_dims.index(inp._ragged_idx)
    inner_dims = [_outer_to_inner_dim(inp_dim, dim) for dim in canonicalized_dims[1:]]
    new_kwargs["dims"] = inner_dims
    return NestedTensor(func(inp._values, **new_kwargs), **inp_kwargs)


@register_jagged_func(
    [torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default],
    "self: jt_all, size: any",
)
def view_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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

    # Preserve inference-mode-ness of input.
    # TODO: Do this for all other views!
    with torch.inference_mode(inp.is_inference()):
        return NestedTensor(func(inp._values, inner_size), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.native_layer_norm.default,
    "input: jt_all, normalized_shape: any, weight: any?, bias: any?, eps: any",
)
def native_layer_norm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    if inp.dim() <= 2:
        raise RuntimeError(
            "layer_norm(): not supported for NestedTensor objects with 2 or fewer dimensions"
        )

    normalized_shape = new_kwargs["normalized_shape"]
    ragged_size = inp.shape[inp._ragged_idx]

    num_dims_not_normalized = inp.dim() - len(normalized_shape)

    if (
        num_dims_not_normalized == 0
    ):  # error if trying to normalize over the batch dimension
        raise RuntimeError(
            "layer_norm(): not supported when normalizing over the batch dimension for NestedTensor"
        )

    if ragged_size in normalized_shape and inp._lengths is not None:
        raise RuntimeError(
            "layer_norm(): not supported where lengths is not None if operating on the ragged dimension for NestedTensor"
        )

    if (
        ragged_size in normalized_shape
    ):  # special handling for normalizing over the ragged dimension
        padded_input = torch.ops.aten._jagged_to_padded_dense_forward(
            inp._values.flatten(
                start_dim=inp._ragged_idx
            ),  # _jagged_to_padded_dense_forward requires values to be a 2D tensor
            [inp._offsets],
            max_lengths=[inp._max_seqlen],  # max length of ragged dimension
        )

        padded_mask = torch.ops.aten._jagged_to_padded_dense_forward(
            torch.ones((inp._values.shape[0], 1), device=inp.device, dtype=inp.dtype),
            [inp._offsets],
            max_lengths=[inp._max_seqlen],  # max length of ragged dimension
        ).expand(
            padded_input.shape
        )  # mask elements outside of the ragged dimension and expand to the same shape as padded input (3D dense tensor)

        ragged_lengths = (
            inp._offsets.diff().unsqueeze(1).unsqueeze(1) * padded_input.shape[2]
        )  # ragged dim * inner dim, since we sum over dims (1, 2) (the layer on which we normalize)

        mean = (
            torch.sum(
                padded_input,
                dim=(1, 2),
                keepdim=True,
            )
            / ragged_lengths
        )  # a sum over (1, 2) ensures layer norm, whereas a sum over (1) would be an instance norm

        padded_normalized = (
            padded_input - mean
        ) * padded_mask  # mask elements outside of the ragged dimension size for correct variance calculation

        variance = (
            torch.sum(
                torch.square(padded_normalized),
                dim=(1, 2),
                keepdim=True,
            )
            / ragged_lengths
        )  # a sum over (1, 2) ensures layer norm, whereas a sum over (1) would be an instance norm

        std = torch.sqrt(variance + new_kwargs["eps"])
        padded_layer_norm = padded_normalized / std

        jagged_layer_norm_values = torch.ops.aten._padded_dense_to_jagged_forward(
            padded_layer_norm,
            [inp._offsets],
            total_L=inp._values.shape[
                0
            ],  # providing this parameter helps avoid a GPU/CPU sync
        ).unflatten(
            -1, inp.shape[inp._ragged_idx + 1 :]
        )  # unflatten last dimension back into original nested tensor shape, e.g. (B, *, WH) --> (B, *, W, H)

        return (
            NestedTensor(jagged_layer_norm_values, **extract_kwargs(inp)),
            mean,
            std,
        )

    output, mean, std = func(inp._values, **new_kwargs)
    return (NestedTensor(output, **extract_kwargs(inp)), mean, std)


@register_jagged_func(
    torch.ops.aten.native_layer_norm_backward.default,
    "grad_out: jt, input: jt, normalized_shape: any, mean: any, rstd: any, weight: any?, bias: any?, output_mask: any",
)
def native_layer_norm_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    grad_out = new_kwargs.pop("grad_out")
    inp = new_kwargs.pop("input")
    d_input, d_gamma, d_beta = func(grad_out._values, inp._values, **new_kwargs)
    if d_input is None:
        return (None, d_gamma, d_beta)

    return (NestedTensor(d_input, **extract_kwargs(inp)), d_gamma, d_beta)


@register_jagged_func(torch.ops.aten.select.int, "self: jt_all, dim: any, index: any")
def select_int(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], "select", allow_batch_dim=True
    )

    # handle batch dim slicing via unbind() for now
    # TODO: make this more efficient
    if new_kwargs["dim"] == 0:
        return inp.unbind()[new_kwargs["index"]]

    if inp._lengths is not None:
        raise ValueError(
            "select(): not yet supported on dim != 0 for non-contiguous nested tensor with holes"
        )

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.slice.Tensor,
    "self: jt, dim: any?, start: any?, end: any?, step: any?",
)
def slice_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    if len(new_kwargs["dim"]) > 1 and (
        inp._ragged_idx in new_kwargs["dim"] or 0 in new_kwargs["dim"]
    ):
        raise RuntimeError(
            "mean(): not supported across multiple dimensions for NestedTensor "
            "when either the batch dim or ragged dim is included"
        )

    (
        new_kwargs["dim"],
        reduce_on_batch,
        reduce_on_ragged,
        _reduce_on_non_batch,
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
        # Note: keepdim=True is on at this point so lengths has to be unsqueezed for
        # that 1-size dim as well.
        lengths = inp._offsets.diff()
        for _ in range(inp.dim() - 1):
            lengths = lengths.unsqueeze(-1)

        return torch_sum / lengths.broadcast_to(torch_sum.shape)

    return NestedTensor(
        func(inp._values, **new_kwargs), **extract_kwargs(inp)
    )  # raggedness preserved


@register_jagged_func(torch.ops.aten.stack.default, "tensors: any, dim: any")
def stack_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # TODO: Handle inference mode properly.
    # See https://github.com/pytorch/pytorch/issues/112024#issuecomment-1779554292
    return inp._values.detach()


@register_jagged_func(torch.ops.aten.all.default, "self: jt_all")
def all_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values)


@register_jagged_func(
    torch.ops.aten.to_padded_tensor.default, "self: jt, padding: any, output_size: any?"
)
def to_padded_tensor_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # TODO: Handle the rest of output_size
    output_size = new_kwargs["output_size"]
    if output_size is not None:
        max_seq_len = output_size[inp._ragged_idx]
    else:
        max_seq_len = inp._max_seqlen

    # only 2D values is supported by the underlying FBGEMM kernel so do shape
    # gymnastics if needed
    values = inp.values()
    values_shape = values.shape
    if values.dim() > 2:
        values = values.flatten(start_dim=1)
    elif values.dim() == 1:
        values = values.unsqueeze(-1)

    # NB: The CUDA kernel for jagged -> padded dense conversion does not support
    # integer / bool types; work around this by casting to half.
    is_bool = values.dtype is torch.bool
    if is_bool and values.is_cuda:
        values = values.to(torch.half)
    padded_out = torch.ops.aten._jagged_to_padded_dense_forward(
        values,
        [inp._offsets],
        [max_seq_len],
        new_kwargs["padding"],
    )
    if is_bool and padded_out.is_cuda:
        padded_out = padded_out.to(torch.bool)

    # shape gymnastics part 2
    if len(values_shape) > 2:
        padded_out = padded_out.unflatten(-1, values_shape[1:])
    elif len(values_shape) == 1:
        padded_out = padded_out.squeeze(-1)

    return padded_out


@register_jagged_func(
    torch.ops.aten._nested_from_padded_tensor.default,
    "padded: t, offsets: t, dummy: jt, ragged_idx: any?, min_seqlen: any?, max_seqlen: any?, sum_S: any?",
)
def _nested_from_padded_tensor_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    if new_kwargs["ragged_idx"] != 1:
        raise RuntimeError(
            "_nested_from_padded_tensor(): only ragged_idx=1 supported for jagged layout"
        )

    padded, offsets = new_kwargs["padded"], new_kwargs["offsets"]

    # non-3D padded is not supported by the underlying FBGEMM kernel so do shape gymnastics
    padded_shape = padded.shape
    if padded.dim() > 3:
        padded = padded.flatten(start_dim=2)
    elif padded.dim() < 3:
        padded = padded.unsqueeze(-1)

    # NB: The CUDA kernel for padded dense -> jagged conversion does not support
    # integer / bool types; work around this by casting to half.
    is_bool = padded.dtype is torch.bool
    if is_bool and padded.is_cuda:
        padded = padded.to(torch.half)
    values = torch.ops.aten._padded_dense_to_jagged_forward(
        padded, [offsets], new_kwargs["sum_S"]
    )
    if is_bool and values.is_cuda:
        values = values.to(torch.bool)

    # shape gymnastics part 2
    if len(padded_shape) > 3:
        values = values.unflatten(-1, padded_shape[2:])
    elif len(padded_shape) < 3:
        values = values.squeeze(-1)

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
        _ragged_idx=ragged_idx,
        _metadata_cache=metadata_cache,
    )


@register_jagged_func(
    torch.ops.aten._nested_view_from_jagged.default,
    "values: t, offsets: t, dummy: jt_all, lengths: t?, ragged_idx: any?, min_seqlen: t?, max_seqlen: t?",
)
def _nested_view_from_jagged_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._offsets


@register_jagged_func(torch.ops.aten._nested_get_lengths.default, "self: jt_all")
def _nested_get_lengths(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._lengths


@register_jagged_func(torch.ops.aten._nested_get_ragged_idx.default, "self: jt_all")
def _nested_get_ragged_idx(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._ragged_idx


@register_jagged_func(torch.ops.aten._nested_get_min_seqlen.default, "self: jt_all")
def _nested_get_min_seqlen(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._metadata_cache.get("min_seqlen", None)


@register_jagged_func(torch.ops.aten._nested_get_max_seqlen.default, "self: jt_all")
def _nested_get_max_seqlen(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    return inp._metadata_cache.get("max_seqlen", None)


# If a section of the Nested Tensor is fully masked out we still retain the section with a length of 0
@register_jagged_func(torch.ops.aten.masked_select.default, "self: jt, mask: any")
def masked_select_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    mask = new_kwargs.pop("mask")

    if inp.ndim > 2:
        raise RuntimeError("masked_select only support 2-D selections currently")
    elif inp.shape != mask.shape:
        raise RuntimeError(
            f"Mask with shape {mask.shape} is not compatible with input's shape {inp.shape}"
        )
    res_values = inp._values.masked_select(mask.values())
    mask_cumsum = F.pad(mask.values().cumsum(dim=0), (1, 0))  # type: ignore[arg-type]

    args = extract_kwargs(inp)
    args["offsets"] = mask_cumsum[inp._offsets]
    return NestedTensor(
        values=res_values,
        **args,
    )


@register_jagged_func(
    torch.ops.aten._nested_select_backward.default,
    "grad_output: t, self: jt, dim: any, index: any",
)
def _nested_select_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    grad_output = new_kwargs.pop("grad_output")

    grad_input = torch.zeros_like(inp, dtype=grad_output.dtype)
    grad_input.select(new_kwargs["dim"], new_kwargs["index"]).copy_(grad_output)

    return grad_input


@register_jagged_func(torch.ops.aten.record_stream.default, "self: jt_all, s: any")
def record_stream_default(func, *args, **kwargs):
    inp = args[0]
    stream = args[1]
    # ensure all components live until stream computation completes
    func(inp._values, stream)
    func(inp._offsets, stream)
    if inp._lengths is not None:
        func(inp._lengths, stream)


@register_jagged_func(
    torch.ops.aten.new_empty.default,
    "self: jt_all, size: any, dtype: any?, layout: any?, device: any?, pin_memory: any?",
)
def new_empty_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    if len(new_kwargs["size"]) == 0:
        return func(inp._values, **new_kwargs)

    raise RuntimeError("new_empty() not supported for NJT with shape != ()")


from torch._higher_order_ops.flex_attention import (
    flex_attention as flex_attention_hop,
    flex_attention_backward as flex_attention_backward_hop,
)
from torch.fx.graph_module import GraphModule


@flex_attention_hop.py_impl(NestedTensor)  # type: ignore[misc]
def flex_njt(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = (),
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4

    # TODO: Support this if needed; determine if NJT buffers need be unwrapped as dense.
    if any(
        isinstance(buf, torch.Tensor) and buf.is_nested
        for buf in score_mod_other_buffers + mask_mod_other_buffers
    ):
        raise RuntimeError(
            "flex_attention(): Nested tensor score_mod / mask_mod buffers are not "
            "currently supported. Please file an issue if this is important to you."
        )

    # need to pass dense tensor of shape (B, n_heads, sum(seq_len), D)
    output = flex_attention_hop(
        query.values().unsqueeze(0),
        key.values().unsqueeze(0),
        value.values().unsqueeze(0),
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        kernel_options=kernel_options,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
    )

    # wrap outputs as NJT
    output_njt = torch.nested.nested_tensor_from_jagged(
        output[0].transpose(1, 2).squeeze(0),
        query._offsets,  # type: ignore[attr-defined]
        query._lengths,  # type: ignore[attr-defined]
        min_seqlen=query._maybe_min_seqlen,  # type: ignore[attr-defined]
        max_seqlen=query._maybe_max_seqlen,  # type: ignore[attr-defined]
    ).transpose(1, 2)

    logsumexp_njt = torch.nested.nested_tensor_from_jagged(
        output[1].transpose(1, 2).squeeze(0),
        query._offsets,  # type: ignore[attr-defined]
        query._lengths,  # type: ignore[attr-defined]
        min_seqlen=query._maybe_min_seqlen,  # type: ignore[attr-defined]
        max_seqlen=query._maybe_max_seqlen,  # type: ignore[attr-defined]
    ).transpose(1, 2)

    return (output_njt, logsumexp_njt)


@flex_attention_backward_hop.py_impl(NestedTensor)  # type: ignore[misc]
def flex_njt_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Union[Callable, GraphModule],
    joint_graph: GraphModule,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple = (),
    mask_mod_other_buffers: Tuple = (),
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], ...]
]:
    output = flex_attention_backward_hop(
        query.values().unsqueeze(0),
        key.values().unsqueeze(0),
        value.values().unsqueeze(0),
        out=out.values().unsqueeze(0),
        logsumexp=logsumexp.values().unsqueeze(0),
        grad_out=grad_out.values().unsqueeze(0),
        grad_logsumexp=grad_logsumexp.values().unsqueeze(0),
        fw_graph=fw_graph,
        joint_graph=joint_graph,
        block_mask=block_mask,
        scale=scale,
        kernel_options=kernel_options,
        score_mod_other_buffers=score_mod_other_buffers,
        mask_mod_other_buffers=mask_mod_other_buffers,
    )

    # wrap grads as NJTs
    dense_q_grad, dense_k_grad, dense_v_grad, score_mod_other_buffer_grads = output
    njt_q_grad = torch.nested.nested_tensor_from_jagged(
        dense_q_grad.transpose(1, 2).squeeze(0),
        query._offsets,  # type: ignore[attr-defined]
        query._lengths,  # type: ignore[attr-defined]
        min_seqlen=query._maybe_min_seqlen,  # type: ignore[attr-defined]
        max_seqlen=query._maybe_max_seqlen,  # type: ignore[attr-defined]
    ).transpose(1, 2)
    njt_k_grad = torch.nested.nested_tensor_from_jagged(
        dense_k_grad.transpose(1, 2).squeeze(0),
        key._offsets,  # type: ignore[attr-defined]
        key._lengths,  # type: ignore[attr-defined]
        min_seqlen=key._maybe_min_seqlen,  # type: ignore[attr-defined]
        max_seqlen=key._maybe_max_seqlen,  # type: ignore[attr-defined]
    ).transpose(1, 2)
    njt_v_grad = torch.nested.nested_tensor_from_jagged(
        dense_v_grad.transpose(1, 2).squeeze(0),
        value._offsets,  # type: ignore[attr-defined]
        value._lengths,  # type: ignore[attr-defined]
        min_seqlen=value._maybe_min_seqlen,  # type: ignore[attr-defined]
        max_seqlen=value._maybe_max_seqlen,  # type: ignore[attr-defined]
    ).transpose(1, 2)

    return (njt_q_grad, njt_k_grad, njt_v_grad, score_mod_other_buffer_grads)


# Make the dummy available on the C++ side.
@register_jagged_func(torch.ops.aten._nested_get_jagged_dummy.default, "self: any")
def _nested_get_jagged_dummy(func, *args, **kwargs):
    from torch.nested._internal.nested_tensor import _nt_view_dummy

    return _nt_view_dummy()


with torch.library._scoped_library("aten", "IMPL") as aten:
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CPU")
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "CUDA")
    aten.impl("_nested_get_jagged_dummy", _nested_get_jagged_dummy, "Meta")

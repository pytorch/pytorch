# mypy: allow-untyped-defs
import functools
import math
import operator
from typing import *  # noqa: F403
from typing import Optional

import torch
import torch.nn.functional as F
from torch.fx.operator_schemas import normalize_function
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention

from .nested_tensor import NestedTensor


__all__: list[Any] = []

JAGGED_OPS_TABLE: Dict[Any, Any] = {}


def _outer_to_inner_dim(ndim, dim, ragged_dim, canonicalize=False):
    from torch._prims_common import canonicalize_dims

    if isinstance(dim, (tuple, list)):
        output = type(dim)(_outer_to_inner_dim(ndim, d, ragged_dim) for d in dim)
        # ensure no duplicates, which can result from both batch and ragged mapping to 0
        return type(output)(dict.fromkeys(output))

    if canonicalize:
        dim = canonicalize_dims(ndim, dim)

    assert dim >= 0 and dim < ndim

    # Map dim=0 (AKA batch dim) -> packed dim i.e. outer ragged dim - 1.
    # For other dims, subtract 1 to convert to inner space.
    return ragged_dim - 1 if dim == 0 else dim - 1


def _wrap_jagged_dim(
    ndim,
    dim,
    ragged_dim,
    op_name,
    convert_to_inner_dim=True,
    allow_ragged_dim=False,
    allow_batch_dim=False,
):
    from torch._prims_common import canonicalize_dims

    wrapped = canonicalize_dims(ndim, dim)
    if wrapped == ragged_dim and not allow_ragged_dim:
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on ragged dim")
    elif wrapped == 0 and not allow_batch_dim:
        raise RuntimeError(f"{op_name}(): not supported for NestedTensor on dim=0")
    ret = (
        _outer_to_inner_dim(ndim, wrapped, ragged_dim)
        if convert_to_inner_dim
        else wrapped
    )
    if allow_batch_dim:
        # Need to disambiguate whether we're operating on the batch dim or not.
        # Operating on dim=1 -> dim=0 after the inner dim conversion.
        operating_on_batch = wrapped == 0
        return (ret, operating_on_batch)
    return ret


def _wrap_jagged_dims(ndim, dims, op_name, ragged_idx=1):
    """
    For NestedTensor operators,
    wraps dimensions to non-negative values,
    and returns metadata related to reduction dimension(s).
    """
    from torch._prims_common import canonicalize_dims

    assert isinstance(dims, (tuple, list)), (
        f"_wrap_jagged_dims(): cannot iterate over dimensions of type {type(dims)}"
    )

    wrapped_dims = [
        canonicalize_dims(ndim, d) for d in dims
    ]  # convert all indices to non-negative values

    operate_on_batch = 0 in wrapped_dims
    operate_on_ragged = ragged_idx in wrapped_dims
    operate_on_non_batch = any(d != 0 and d != ragged_idx for d in wrapped_dims)

    # ensure no duplicates, which can result from both batch and ragged mapping to 0
    outer_to_inner_dim = tuple(
        dict.fromkeys(_outer_to_inner_dim(ndim, d, ragged_idx) for d in wrapped_dims)
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
            inp.dim(),
            new_kwargs["start_dim"],
            inp._ragged_idx,
            "flatten",
            convert_to_inner_dim=False,
        )
        end_dim = _wrap_jagged_dim(
            inp.dim(),
            new_kwargs["end_dim"],
            inp._ragged_idx,
            "flatten",
            convert_to_inner_dim=False,
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
            assert inp._ragged_idx == 1, (
                "NJT with ragged_idx != 1 not supported for contiguous clone"
            )
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
        # NB: Fold dims of values for input and grad_output to treat them as 2D. This
        # trick avoids materializing large intermediates and immediately reducing over
        # them via sum(). This is equivalent to computing:
        #     torch.matmul(grad_output._values.transpose(-2, -1), inp._values)
        # and then summing over the leading dimensions to get a 2D weight grad.
        grad_2d = grad_output._values.reshape(-1, weight.size(0))
        input_2d = inp._values.reshape(-1, weight.size(1))
        dw = torch.matmul(grad_2d.t(), input_2d)
    if output_mask[2]:
        # Sum over all but the last dim to get a 1D bias grad. We cannot
        # rely on the autograd engine to reduce for us, because returning a
        # tensor aliasing the input would violate the aten signature annotation
        reduce_dims = tuple(range(grad_output._values.ndim - 1))
        if reduce_dims == ():
            db = grad_output._values.clone()
        else:
            db = torch.sum(grad_output._values, reduce_dims, keepdim=False)
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
        # try to recursively copy_ on unbound components to get around nested int mismatch
        # TODO: eventually do a direct copy when this is possible
        inp_comps = inp.unbind()
        inp_comp_shapes = [c.shape for c in inp_comps]
        src_comps = src.unbind()
        src_comp_shapes = [c.shape for c in src_comps]
        if inp_comp_shapes != src_comp_shapes:
            raise RuntimeError(
                "copy_(): expected compatible input and src shapes, but got: "
                f"{inp.shape} and {src.shape}"
            )
        for inp_comp, src_comp in zip(inp_comps, src_comps):
            inp_comp.copy_(src_comp)

    # AOTD allows mutations of inputs only, (not views of the inputs).
    # NJT.values() returns _values.detach() to workaround some issues.
    # To keep mutation in the graph, AOTD manually calls copy_ on the input (NJT).
    # Here we directly mutate self._values to not emit .detach() in the graph, which would make it non-compilable.
    inp._values.copy_(src._values)
    return inp


register_jagged_func(torch.ops.aten.detach.default, "self: jt_all")(
    jagged_unary_pointwise
)


@register_jagged_func(
    [
        torch.ops.aten.empty_like.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.rand_like.default,
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

    new_values = func(inp._values, **new_kwargs)
    new_offsets = inp._offsets.to(device=new_values.device)
    new_lengths = None
    if inp._lengths is not None:
        new_lengths = inp._lengths.to(device=new_values.device)
    output_kwargs = extract_kwargs(inp)
    if "offsets" in output_kwargs:
        output_kwargs["offsets"] = new_offsets
    if "lengths" in output_kwargs:
        output_kwargs["lengths"] = new_lengths

    if inp.device != new_values.device:
        # Update the nested int registry to indicate that the ragged structure is the same
        # between the two offsets / lengths on different devices.
        from torch._subclasses.fake_tensor import FakeTensor
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            mb_unwrap_functional_tensor,
        )

        from .nested_tensor import _tensor_symint_registry

        ragged_source = inp._offsets if inp._lengths is None else inp._lengths
        new_thing = new_offsets if new_lengths is None else new_lengths
        if isinstance(new_thing, (FakeTensor, FunctionalTensor)):
            # Temporary hack until we have the union find
            tgt = mb_unwrap_functional_tensor(new_thing)
            src = mb_unwrap_functional_tensor(ragged_source)
            tgt.nested_int_memo = src.nested_int_memo
        else:
            _tensor_symint_registry[new_thing] = _tensor_symint_registry[ragged_source]

    return NestedTensor(new_values, **output_kwargs)


register_jagged_func(torch.ops.aten.full_like.default, "self: jt_all, fill_value: any")(
    like_factory_default
)

register_jagged_func(torch.ops.aten.randint_like.default, "self: jt_all, high: any")(
    like_factory_default
)

register_jagged_func(
    torch.ops.aten.randint_like.low_dtype, "self: jt_all, low: any, high: any"
)(like_factory_default)


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


@register_jagged_func(
    torch.ops.aten.prod.dim_int,
    "self: jt_all, dim: any, keepdim: any?, dtype: any?",
)
def prod_dim_int(func, *args, **kwargs):
    return _apply_reduction(func, "prod", 1, *args, **kwargs)


@register_jagged_func(torch.ops.aten.prod.default, "self: jt_all, dtype: any?")
def prod_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(
    torch.ops.aten.split.Tensor, "self: jt, split_size: any, dim: any?"
)
def split_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "split"
    )

    return tuple(
        NestedTensor(values=x, **extract_kwargs(inp))
        for x in func(inp._values, **new_kwargs)
    )


@register_jagged_func(
    torch.ops.aten.split_with_sizes.default, "self: jt, split_sizes: any, dim: any?"
)
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "split_with_sizes"
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

    dim = _wrap_jagged_dim(inp.dim(), new_kwargs["dim"], inp._ragged_idx, "narrow")
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

    new_kwargs["dim"], operating_on_batch = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "chunk", allow_batch_dim=True
    )

    if operating_on_batch:
        chunks = new_kwargs["chunks"]

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

        # Note that the actual number of chunks returned is not necessarily the same as
        # the input number; it can be counter-intuitive, but it matches dense behavior.
        return [
            NestedTensor(values=chunk_values[i], **(nested_kwargs[i]))
            for i in range(0, len(chunk_values))
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

    def _torch_check(_lengths: list[int], _offsets: Optional[list[int]] = None):
        # This torch._check and torch._check_is_size are needed for torch.compile
        # symbolic shapes processing.
        # offsets and lengths are symbolic variables during compilation,
        # we guarantee the correct offsets/lengths correspondence:
        # sum of lengths <= total ragged_dim_size
        # every length and offset are size-like variable (allows sym shapes to reason it as [2, inf))
        # offset[i] + length[i] <= ragged_dim_size, for unbind and split dim correctness
        # offsets[i] <= ragged_dim_size

        lengths_sum = 0
        ragged_dim_size = values.shape[ragged_idx - 1]
        for i in range(len(_lengths)):
            torch._check_is_size(_lengths[i])
            torch._check(_lengths[i] <= ragged_dim_size)

            lengths_sum += _lengths[i]
            if _offsets is not None:
                torch._check(
                    _offsets[i] + _lengths[i] <= ragged_dim_size,
                    lambda: "unbind(): nested tensor offsets and lengths do not match ragged_idx dimension",
                )
        torch._check(lengths_sum <= ragged_dim_size)

        if _offsets is not None:
            for i in range(len(_offsets)):
                torch._check_is_size(_offsets[i])
                torch._check(_offsets[i] <= ragged_dim_size)

    if lengths is None:
        lengths_scalars = offsets.diff().tolist()
        _torch_check(lengths_scalars)

        return torch.split(values, lengths_scalars, dim=(ragged_idx - 1))

    if ragged_idx <= 0:
        raise RuntimeError(
            "unbind(): nested tensor ragged_idx out of bounds (should be >= 1)"
        )

    lengths_scalars = lengths.tolist()
    offsets_scalars = offsets.tolist()

    _torch_check(lengths_scalars, offsets_scalars)

    return [
        torch.narrow(
            values,
            dim=(ragged_idx - 1),
            start=offsets_scalars[i],
            length=lengths_scalars[i],
        )
        for i in range(lengths.shape[0])
    ]


@register_jagged_func(torch.ops.aten.squeeze.dim, "self: jt, dim: any")
def squeeze_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    new_kwargs["dim"] = _wrap_jagged_dim(
        len(inp._size), new_kwargs["dim"], inp._ragged_idx, "squeeze"
    )
    return NestedTensor(func(values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.unsqueeze.default, "self: jt_all, dim: any")
def unsqueeze_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    values = inp._values

    # Account for collapsed jagged dim
    dim = new_kwargs["dim"]
    new_kwargs["dim"] = _wrap_jagged_dim(
        len(inp._size) + 1, dim, inp._ragged_idx, "unsqueeze", allow_ragged_dim=True
    )

    # ragged_idx changes if a dimension is added before it
    output_kwargs = extract_kwargs(inp)
    if new_kwargs["dim"] <= inp._ragged_idx - 1:
        output_kwargs["_ragged_idx"] += 1

    return NestedTensor(func(values, **new_kwargs), **output_kwargs)


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
    new_kwargs["dim"] = _wrap_jagged_dim(
        len(first.shape), dim, first._ragged_idx, "cat"
    )

    return NestedTensor(
        func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0])
    )


@register_jagged_func(torch.ops.aten.matmul.default, "self: any, other: any")
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
        if a.is_nested:
            nt = a
        else:
            nt = b

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
        if a.is_nested:
            padded_t = func(padded_nt, b)
        else:
            padded_t = func(a, padded_nt)
        return nested_from_padded(
            padded_t,
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
        if (
            inp.dim() >= 3
            and inp.dim() == other.dim()
            and inp._ragged_idx < inp.dim() - 1
        ):
            # convert to padded for this
            return _padded_impl(inp, other)
        # Support broadcasting the dense:
        # (B, j1, D) x (D, E) => (B, j1, E)
        # (B, j1, D, E) x (E, F) => (B, j1, D, F)
        # etc.
        elif (
            other.dim() == 2
            and inp.dim() > other.dim()
            and inp._ragged_idx < inp.dim() - 1
        ):
            return NestedTensor(
                func(inp._values, other, **new_kwargs), **extract_kwargs(inp)
            )
    # Dense x NJT
    elif not inp.is_nested and other.is_nested:
        # (B, D, E) x (B, E, j1) => (B, E, j1)
        if other.dim() >= 3 and other.dim() == inp.dim() and other._ragged_idx >= 2:
            # convert to padded for this
            return _padded_impl(inp, other)
        # Support broadcasting the dense:
        # (D, E) x (B, E, j1) => (B, D, j1)
        # (D, E) x (B, E, j1, F) => (B, D, j1, F)
        # etc.
        elif inp.dim() == 2 and other.dim() > inp.dim() and other._ragged_idx >= 2:
            return NestedTensor(
                func(inp, other._values, **new_kwargs), **extract_kwargs(other)
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
    torch.ops.aten.expand.default, "self: jt_all, size: any, implicit: any?"
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

    expand_arg = [-1 if d == inp._ragged_idx else size[d] for d in range(1, inp.dim())]
    return NestedTensor(func(inp._values, expand_arg), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.expand_as.default, "self: t, other: jt")
def expand_as_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    return NestedTensor(func(inp, other._values), **extract_kwargs(other))


@register_jagged_func(torch.ops.aten.broadcast_to.default, "self: jt_all, size: any")
def broadcast_to(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    size = new_kwargs.pop("size")

    if len(size) <= inp.dim():
        return inp.expand([*(1 for _ in range(inp.dim() - len(size))), *size])

    raise ValueError(
        "broadcast_to(): broadcasting to a higher-dim shape is currently not supported "
        "for nested tensors with the jagged layout"
    )


@register_jagged_func(torch.ops.aten.broadcast_tensors.default, "tensors: any")
def broadcast_tensors(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    tensors = new_kwargs.pop("tensors")
    if len(tensors) == 0:
        raise ValueError("broadcast_tensors(): expected at least one tensor input")
    if len(tensors) == 1:
        return tensors[0]

    outs = []
    broadcast_shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    # Pull out the first NJT. If broadcast_shapes() worked, the nested ints are compatible.
    njt = next(t for t in tensors if isinstance(t, NestedTensor))
    for t in tensors:
        if t.is_nested:
            outs.append(t.broadcast_to(broadcast_shape))
        elif t.dim() < len(broadcast_shape):
            outs.append(
                NestedTensor(t.broadcast_to(njt._values.shape), **extract_kwargs(njt))
            )
        else:
            raise ValueError(
                "broadcast_tensors(): broadcasting nested tensors with dense tensors of equal "
                "or higher dim is not currently supported"
            )

    return tuple(outs)


@register_jagged_func(
    torch.ops.aten.where.self, "condition: jt_all, self: any, other: any"
)
def where_self(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    condition = new_kwargs.pop("condition")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")

    # if the tensors aren't compatible, broadcast_tensors() will let us know
    condition, inp, other = torch.broadcast_tensors(condition, inp, other)

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


def _apply_reduction(func, func_name, identity_element, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    # some ops use dim=None to indicate a full reduction; some use an empty dim list
    full_reduction = new_kwargs["dim"] is None or (
        isinstance(new_kwargs["dim"], (tuple, list)) and len(new_kwargs["dim"]) == 0
    )
    if full_reduction:
        out = func(inp._values, **new_kwargs)
        if new_kwargs.get("keepdim", False):
            if isinstance(out, (tuple, list)):
                # some ops return multiple things; unsqueeze all of them
                out = type(out)(o.unsqueeze(inp._ragged_idx) for o in out)
            else:
                out = out.unsqueeze(inp._ragged_idx)
        return out

    # some ops support lists of dims; some don't
    dim_to_convert = new_kwargs["dim"]
    is_dimlist = isinstance(new_kwargs["dim"], (tuple, list))
    if not is_dimlist:
        dim_to_convert = [dim_to_convert]

    (
        converted_dim,
        reduce_on_batch,
        reduce_on_ragged,
        reduce_on_non_batch,
    ) = _wrap_jagged_dims(
        inp.dim(),
        dim_to_convert,
        f"{func_name}",
        inp._ragged_idx,
    )

    if not is_dimlist:
        # convert back from list
        converted_dim = converted_dim[0]
    new_kwargs["dim"] = converted_dim

    if reduce_on_ragged and inp._lengths is not None:
        raise RuntimeError(
            f"{func_name}(): reducing across the ragged dimension is not supported "
            "for non-contiguous nested tensors with holes"
        )

    from torch.utils._pytree import tree_map

    # raggedness reduced away --> return dense tensor
    if reduce_on_ragged:
        # reduction cases: (batch, ragged), (batch, ragged, non-batch), etc.
        if reduce_on_batch:
            # no need to read offsets --> apply sum directly on values
            out = func(inp._values, **new_kwargs)
            if new_kwargs.get("keepdim", False):
                # some ops return multiple things; unsqueeze all of them
                out = tree_map(lambda o: o.unsqueeze(0), out)
            return out
        else:
            # invalid reduction cases: (ragged, non-batch), etc.
            if reduce_on_non_batch:
                raise RuntimeError(
                    f"{func_name}(): reducing along a ragged and non-batch dimension "
                    "is not supported for nested tensors"
                )

            # reduction cases: (ragged)
            # convert to padded dense and reduce
            new_kwargs.pop("dim")
            dim_to_pass = [inp._ragged_idx] if is_dimlist else inp._ragged_idx
            return func(
                inp.to_padded_tensor(identity_element), dim=dim_to_pass, **new_kwargs
            )
    # raggedness preserved --> return nested tensor
    else:
        # invalid reduction cases: (batch), (batch, non-batch), etc.
        if reduce_on_batch:
            raise RuntimeError(
                f"{func_name}(): reducing along the batch dimension but not "
                "the ragged dimension is not supported for nested tensors"
            )

        # reduction cases: (non-batch), (non-batch, non-batch), etc.
        # apply sum directly on values
        out = func(inp._values, **new_kwargs)
        out_kwargs = extract_kwargs(inp)
        if not new_kwargs.get("keepdim", False):
            # dims are reduced away -> ragged_idx of output needs to be reevaluated
            dimlist = (
                new_kwargs["dim"]
                if isinstance(new_kwargs["dim"], (tuple, list))
                else [new_kwargs["dim"]]
            )
            for d in dimlist:
                # adjust for all dims reduced before the ragged dim
                if d < inp._ragged_idx - 1:
                    out_kwargs["_ragged_idx"] -= 1

        # some ops return multiple things; wrap each of them as an NJT
        return tree_map(lambda o: NestedTensor(o, **out_kwargs), out)


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
    return _apply_reduction(func, "sum", 0, *args, **kwargs)


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
                _outer_to_inner_dim(len(inp._size), dim0, inp._ragged_idx),
                _outer_to_inner_dim(len(inp._size), dim1, inp._ragged_idx),
            ),
            **inp_kwargs,
        )

    new_kwargs["dim0"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim0"], inp._ragged_idx, "transpose"
    )
    new_kwargs["dim1"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim1"], inp._ragged_idx, "transpose"
    )

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
    inner_dims = [
        _outer_to_inner_dim(inp_dim, dim, inp._ragged_idx)
        for dim in canonicalized_dims[1:]
    ]
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
            (padded_input - mean) * padded_mask
        )  # mask elements outside of the ragged dimension size for correct variance calculation

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
    new_kwargs["dim"], operating_on_batch = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "select", allow_batch_dim=True
    )

    # handle batch dim slicing via unbind() for now
    # TODO: make this more efficient
    if operating_on_batch:
        return inp.unbind()[new_kwargs["index"]]

    if inp._lengths is not None:
        raise ValueError(
            "select(): not yet supported on dim != 0 for non-contiguous nested tensor with holes"
        )

    # if selecting before the ragged dim, adjust output ragged_idx
    out_kwargs = extract_kwargs(inp)
    if new_kwargs["dim"] < inp._ragged_idx - 1:
        out_kwargs["_ragged_idx"] -= 1

    return NestedTensor(func(inp._values, **new_kwargs), **out_kwargs)


@register_jagged_func(
    torch.ops.aten.slice.Tensor,
    "self: jt, dim: any?, start: any?, end: any?, step: any?",
)
def slice_tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    new_kwargs["dim"] = _wrap_jagged_dim(
        inp.dim(), new_kwargs["dim"], inp._ragged_idx, "slice"
    )

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(
    torch.ops.aten.index_put.default,
    "input: jt_all, indices: any, values: t, accumulate: any?",
)
@register_jagged_func(
    torch.ops.aten.index_put_.default,
    "input: jt_all, indices: any, values: t, accumulate: any?",
)
def index_put_(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp: NestedTensor = new_kwargs.pop("input")

    # For index_put_ to work, we add together the indices of the ragged dimension
    # and the batch dimension, adding the offsets of each ragged dimension to its
    # indices

    indices = new_kwargs.pop("indices")

    assert len(indices) <= inp.dim()

    if len(indices) < inp._ragged_idx + 1:
        if not inp.is_contiguous():
            raise RuntimeError(
                "index_put(): If ragged dimension is not part of indices, this only works on contiguous NJTs"
            )
        # Ragged dim is NOT part of indices, we need to pad the nested tensor to apply func
        from .nested_tensor import nested_from_padded

        min_seqlen = inp._maybe_min_seqlen
        max_seqlen = inp._maybe_max_seqlen
        padded_max_S = max_seqlen
        total_L = inp._values.shape[inp._ragged_idx - 1]
        if padded_max_S is None:
            # use upper bound on max seqlen if it's not present
            padded_max_S = total_L

        padded_shape = (
            *inp.shape[: inp._ragged_idx],
            padded_max_S,
            *inp.shape[inp._ragged_idx + 1 :],
        )
        padded_inp = inp.to_padded_tensor(0.0, output_size=padded_shape)
        new_njt = nested_from_padded(
            func(padded_inp, indices, **new_kwargs),
            offsets=inp._offsets,
            ragged_idx=inp._ragged_idx,
            sum_S=total_L,
            min_seqlen=min_seqlen,
            max_seqlen=max_seqlen,
        )

        if func == torch.ops.aten.index_put_.default:
            inp._values.copy_(new_njt.values())
            return inp
        return new_njt

    # We can run on the underlying values directly

    # Validate indices
    if inp.lengths() is None:
        lengths = inp.offsets().diff()
    else:
        lengths = inp.lengths()
    torch._assert_async(
        torch.all(indices[inp._ragged_idx] < lengths),
        "Some indices in the ragged dimension are out of bounds!",
    )

    # Recompute indices for _values
    ragged_indices = inp.offsets()[indices[0]] + indices[inp._ragged_idx]
    func_indices = (
        # before ragged dim
        indices[1 : inp._ragged_idx]
        # ragged dim (combined with batch)
        + [ragged_indices]
        # after ragged dim
        + indices[inp._ragged_idx + 1 :]
    )

    if func == torch.ops.aten.index_put_.default:
        inp._values = func(inp._values, func_indices, **new_kwargs)
        return inp

    return NestedTensor(
        func(inp._values, func_indices, **new_kwargs),
        **extract_kwargs(inp),
    )


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
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs["input"]
    (_, reduce_on_batch, reduce_on_ragged, reduce_on_non_batch) = _wrap_jagged_dims(
        inp.dim(),
        new_kwargs["dim"],
        "mean",
        inp._ragged_idx,
    )

    if reduce_on_ragged and not reduce_on_batch:
        assert not reduce_on_non_batch
        # calculate an intermediate sum and leave the dim in for normalization purposes
        keepdim = new_kwargs["keepdim"]
        new_kwargs["keepdim"] = True
        intermediate_sum = _apply_reduction(
            torch.ops.aten.sum.dim_IntList, "mean", 0, **new_kwargs
        )

        # normalize by sequence lengths
        lengths = inp._lengths if inp._lengths is not None else inp._offsets.diff()
        for _ in range(intermediate_sum.dim() - 1):
            lengths = lengths.unsqueeze(-1)
        out = intermediate_sum / lengths
        if not keepdim:
            out = out.squeeze(inp._ragged_idx)
        return out

    # at this point, we're just redispatching on the values buffer
    # since we expect it to be unused, specify a weird intermediate value to
    # hopefully make errors obvious
    intermediate_value = 0.42
    return _apply_reduction(func, "mean", intermediate_value, **new_kwargs)


@register_jagged_func(torch.ops.aten.mean.default, "self: jt_all, dtype: any?")
def mean_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(torch.ops.aten.any.dims, "self: jt_all, dim: any?, keepdim: any?")
def any_dims(func, *args, **kwargs):
    return _apply_reduction(func, "any", False, *args, **kwargs)


@register_jagged_func(torch.ops.aten.any.dim, "self: jt_all, dim: any, keepdim: any?")
def any_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # wrap dim in list to redispatch to dims overload
    new_kwargs["dim"] = [new_kwargs["dim"]]
    return any_dims(torch.ops.aten.any.dims, **new_kwargs)


@register_jagged_func(torch.ops.aten.all.dims, "self: jt_all, dim: any?, keepdim: any?")
def all_dims(func, *args, **kwargs):
    return _apply_reduction(func, "all", True, *args, **kwargs)


@register_jagged_func(torch.ops.aten.all.dim, "self: jt_all, dim: any, keepdim: any?")
def all_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    # wrap dim in list to redispatch to dims overload
    new_kwargs["dim"] = [new_kwargs["dim"]]
    return all_dims(torch.ops.aten.all.dims, **new_kwargs)


@register_jagged_func(
    [
        torch.ops.aten.all.default,
        torch.ops.aten.any.default,
        torch.ops.aten.max.default,
        torch.ops.aten.min.default,
    ],
    "self: jt_all",
)
def all_any_max_min_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return func(inp._values, **new_kwargs)


@register_jagged_func(torch.ops.aten.min.dim, "self: jt_all, dim: any, keepdim: any?")
def min_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype_max = torch.finfo(new_kwargs["input"].dtype).max
    return _apply_reduction(func, "min", dtype_max, *args, **kwargs)


@register_jagged_func(torch.ops.aten.max.dim, "self: jt_all, dim: any, keepdim: any?")
def max_dim(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype_min = torch.finfo(new_kwargs["input"].dtype).min
    return _apply_reduction(func, "max", dtype_min, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.amin.default, "self: jt_all, dim: any?, keepdim: any?"
)
def amin_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype_max = torch.finfo(new_kwargs["input"].dtype).max
    return _apply_reduction(func, "amin", dtype_max, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.amax.default, "self: jt_all, dim: any?, keepdim: any?"
)
def amax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype_min = torch.finfo(new_kwargs["input"].dtype).min
    return _apply_reduction(func, "amax", dtype_min, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.argmin.default, "self: jt_all, dim: any?, keepdim: any?"
)
def argmin_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype_max = torch.finfo(new_kwargs["input"].dtype).max
    return _apply_reduction(func, "argmin", dtype_max, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.argmax.default, "self: jt_all, dim: any?, keepdim: any?"
)
def argmax_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    dtype_min = torch.finfo(new_kwargs["input"].dtype).min
    return _apply_reduction(func, "argmax", dtype_min, *args, **kwargs)


@register_jagged_func(
    torch.ops.aten.value_selecting_reduction_backward.default,
    "grad: jt_all, dim: any, indices: jt_all, sizes: any, keepdim: any",
)
def value_selecting_reduction_backward_default(func, *args, **kwargs):
    from torch.fx.experimental.symbolic_shapes import is_nested_int

    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    grad = new_kwargs.pop("grad")
    new_kwargs["grad"] = grad._values
    indices = new_kwargs.pop("indices")
    new_kwargs["indices"] = indices._values
    # should always succeed; sizes should contain a nested int
    ragged_idx = next(i for i, s in enumerate(new_kwargs["sizes"]) if is_nested_int(s))
    # convert dim -> values-space dim
    new_kwargs["dim"] = _wrap_jagged_dim(
        len(new_kwargs["sizes"]),
        new_kwargs["dim"],
        ragged_idx,
        "value_selecting_reduction_backward",
    )
    # convert saved NJT sizes -> values-space sizes
    sizes = new_kwargs.pop("sizes")
    sizes[ragged_idx] = indices._values.size(indices._ragged_idx - 1)
    sizes = sizes[1:]
    new_kwargs["sizes"] = sizes

    output_kwargs = extract_kwargs(indices)
    output_kwargs["_ragged_idx"] = ragged_idx

    return NestedTensor(func(**new_kwargs), **output_kwargs)


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
        tensors[0].dim() + 1, new_kwargs["dim"], tensors[0]._ragged_idx, "stack"
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
    torch.ops.aten.embedding_dense_backward.default,
    "grad_output: jt, indices: jt, num_weights: any, padding_idx: any, scale_grad_by_freq: any",
)
def embedding_dense_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    indices = new_kwargs.pop("indices")
    grad_output = new_kwargs.pop("grad_output")
    return func(grad_output._values, indices._values, **new_kwargs)


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
    torch.ops.aten.to_padded_tensor.default,
    "self: jt_all, padding: any, output_size: any?",
)
def to_padded_tensor_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    if inp._lengths is not None:
        raise RuntimeError(
            "to_padded_tensor(): not supported for nested tensors with holes"
        )

    # TODO: Handle the rest of output_size
    output_size = new_kwargs["output_size"]
    if output_size is not None:
        max_seq_len = output_size[inp._ragged_idx]
    else:
        max_seq_len = (
            inp._max_seqlen
            if inp._max_seqlen_tensor is not None
            else inp._values.size(0)
        )

    # only 2D values with ragged packed dim=0 is supported by the underlying FBGEMM
    # kernel so do shape gymnastics if needed
    values = inp.values()
    if inp._ragged_idx > 1:
        values = values.transpose(inp._ragged_idx - 1, 0)
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
    if inp._ragged_idx > 1:
        padded_out = padded_out.transpose(inp._ragged_idx, 1)

    return padded_out


@register_jagged_func(
    torch.ops.aten._nested_from_padded_tensor.default,
    "padded: t, offsets: t, dummy: jt, ragged_idx: any?, min_seqlen: any?, max_seqlen: any?, sum_S: any?",
)
def _nested_from_padded_tensor_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    padded, offsets = new_kwargs["padded"], new_kwargs["offsets"]
    ragged_idx = new_kwargs.get("ragged_idx", 1)

    # only 3D padded with ragged packed dim=0 is supported by the underlying FBGEMM
    # kernel so do shape gymnastics
    if ragged_idx > 1:
        padded = padded.transpose(ragged_idx, 1)
    padded_ragged_dim1_shape = padded.shape
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
    if len(padded_ragged_dim1_shape) > 3:
        values = values.unflatten(-1, padded_ragged_dim1_shape[2:])
    elif len(padded_ragged_dim1_shape) < 3:
        values = values.squeeze(-1)
    if ragged_idx > 1:
        values = values.transpose(ragged_idx - 1, 0)

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
    "grad_output: t, self: jt_all, dim: any, index: any",
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
    [
        torch.ops.aten.new_empty.default,
        torch.ops.aten.new_zeros.default,
        torch.ops.aten.new_ones.default,
    ],
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


@register_jagged_func(
    [
        torch.ops.aten.elu_backward.default,
        torch.ops.aten.hardshrink_backward.default,
        torch.ops.aten.hardsigmoid_backward.default,
        torch.ops.aten.hardtanh_backward.default,
        torch.ops.aten.softplus_backward.default,
        torch.ops.aten.softshrink_backward.default,
    ],
    "self: jt_all, ...",
)
def activation_backward(func, *args, **kwargs):
    # first NJT arg is expected to be grad_output
    grad_output = next(arg for arg in args if isinstance(arg, NestedTensor))
    return NestedTensor(
        func(
            *(arg._values if isinstance(arg, NestedTensor) else arg for arg in args),
            **kwargs,
        ),
        **extract_kwargs(grad_output),
    )


@register_jagged_func(torch.ops.aten.fill.Scalar, "self: jt_all, value: any")
def fill_Scalar(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))


@register_jagged_func(torch.ops.aten.fill_.Scalar, "self: jt_all, value: any")
def fill__Scalar(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")

    func(inp._values, **new_kwargs)
    return inp


@register_jagged_func(torch.ops.aten.frexp.Tensor, "self: jt_all")
def frexp_Tensor(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    inp = new_kwargs.pop("input")
    output_kwargs = extract_kwargs(inp)

    mantissa, exponent = func(inp._values)
    return NestedTensor(mantissa, **output_kwargs), NestedTensor(
        exponent, **output_kwargs
    )


@register_jagged_func(
    torch.ops.aten.matmul_backward.default,
    "grad: any, self: any, other: any, mask: any",
)
def matmul_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    grad = new_kwargs.pop("grad")
    inp = new_kwargs.pop("input")
    other = new_kwargs.pop("other")
    grad_input_mask = new_kwargs.pop("mask")

    if grad is None:
        return (None, None)

    grad_self = None
    if grad_input_mask[0]:
        grad_self = torch.matmul(grad, other.transpose(-1, -2))

    grad_other = None
    if grad_input_mask[1]:
        grad_other = torch.matmul(inp.transpose(-1, -2), grad)

    return (grad_self, grad_other)


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

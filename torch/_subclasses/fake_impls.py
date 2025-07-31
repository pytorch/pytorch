# mypy: ignore-errors

import functools
import itertools
import math
import operator
import sys
from functools import reduce
from typing import Callable, Union

import torch
import torch._custom_op
import torch._logging
import torch._prims_common as utils
from torch._dispatch.python import no_python_dispatcher
from torch._ops import OpOverload
from torch._prims_common import (
    contiguous_for_memory_format_or_false,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_boolean_dtype,
    is_contiguous,
    is_contiguous_or_false,
    is_float_dtype,
    is_integer_dtype,
    make_contiguous_strides_for,
)
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensor,
    in_kernel_invocation_manager,
    run_fallback_kernel,
    UnsupportedOperatorException,
)
from torch.fx.operator_schemas import normalize_function
from torch.utils._stats import count_label


pytree = torch.utils._pytree

__all__ = [
    "op_implementations_checks",
    "get_fast_op_impls",
    "stride_incorrect_op",
    "has_meta",
]

op_implementations_dict = {}
op_implementations_checks = []


aten = torch._ops.ops.aten


def ordered_set(*items):
    return dict.fromkeys(items, True)


# This function indicates if the backend device
# supports non-contiguous tensors
def is_noncontiguous_supported(device):
    return device.type != "hpu"


_like_tensor_constructors = ordered_set(
    aten.empty_like.default,
    aten.empty_like.out,
    aten.full_like.default,
    aten.full_like.out,
    aten.ones_like.default,
    aten.ones_like.out,
    aten.rand_like.default,
    aten.rand_like.out,
    aten.randn_like.default,
    aten.randn_like.out,
    aten.randint_like.default,
    aten.randint_like.Tensor,
    aten.randint_like.Tensor_out,
    aten.randint_like.out,
    aten.randint_like.low_dtype,
    aten.randint_like.low_dtype_out,
    aten.zeros_like.default,
    aten.zeros_like.out,
    aten.new_empty.default,
    aten.new_empty.out,
    aten.new_empty_strided.default,
    aten.new_empty_strided.out,
    aten.new_full.default,
    aten.new_full.out,
    aten.new_zeros.default,
    aten.new_zeros.out,
    aten.new_ones.default,
    aten.new_ones.out,
)


_device_not_kwarg_ops = ordered_set(
    aten._resize_output_.default,
    aten._nested_tensor_from_tensor_list.default,
    aten._nested_tensor_from_tensor_list.out,
    aten.pin_memory.default,
    aten.to.device,
    aten.to.prim_Device,
    aten.is_pinned.default,
    aten._pin_memory.default,
    aten._pin_memory.out,
    aten._resize_output.default,
    aten._resize_output.out,
)

# this op is never actually used
_non_kwarg_device_constructors = (aten._list_to_tensor,)


def contains_tensor_types(type):
    tensor_type = torch._C.TensorType.get()
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )


@functools.cache
def _is_tensor_constructor(func: OpOverload):
    assert isinstance(func, OpOverload)
    schema = func._schema
    if any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return False
    # TODO: no real reason to restrict multiple outputs
    return (
        len(schema.returns) == 1 and schema.returns[0].type is torch._C.TensorType.get()
    )


def register_op_impl(run_impl_check: Union[Callable[[OpOverload], bool], OpOverload]):
    def impl_decorator(op_impl):
        if isinstance(run_impl_check, OpOverload):
            assert run_impl_check not in op_implementations_dict, (
                f"duplicate registration: {run_impl_check}"
            )
            op_implementations_dict[run_impl_check] = op_impl
        elif isinstance(run_impl_check, (list, tuple)):
            for op in run_impl_check:
                register_op_impl(op)(op_impl)
        else:
            assert callable(run_impl_check)
            op_implementations_checks.append((run_impl_check, op_impl))

        return op_impl

    return impl_decorator


def _is_op_registered_to_fake_rule(op):
    return op in op_implementations_dict


def _deregister_op_impl(op):
    if op in op_implementations_dict:
        del op_implementations_dict[op]
    for check, impl in op_implementations_checks:
        if check is op:
            op_implementations_checks.remove((check, impl))
            break


@register_op_impl(op_implementations_dict.__contains__)
def dispatch_to_op_implementations_dict(fake_mode, func, *args, **kwargs):
    return op_implementations_dict[func](fake_mode, func, *args, **kwargs)


@register_op_impl(_is_tensor_constructor)
@register_op_impl([*_like_tensor_constructors])
def constructors(fake_mode, func, *args, **kwargs):
    assert func not in _non_kwarg_device_constructors
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    if "names" in kwargs:
        raise UnsupportedOperatorException(
            "torch.compile doesn't support named tensors"
        )

    if func in _like_tensor_constructors:
        default_device = new_kwargs["input"].device
        # TODO: file issue
        args = (new_kwargs.pop("input"),)
    else:
        # cpu is default device if none is specified
        default_device = torch.device("cpu")
        args = ()
    out_device = new_kwargs.pop("device", None)
    out_device = out_device if out_device is not None else default_device
    new_kwargs["device"] = torch.device("meta")
    # _like constructors have fake tensor inputs (maybe this causes the non-like
    # to fail? hmmm)
    with in_kernel_invocation_manager(fake_mode):
        r = func(*args, **new_kwargs)
    return FakeTensor(fake_mode, r, out_device)


@register_op_impl(aten.is_pinned.default)
def non_kwarg_is_pinned(fake_mode, func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args, kwargs, normalize_to_only_use_kwargs=True
    )
    inp = new_kwargs.pop("input")
    # we'll ignore device argument because it is deprecated and not
    # actually used by is_pinned.
    with in_kernel_invocation_manager(fake_mode):
        r = func(inp)
    return r


@register_op_impl(aten.to.prim_Device)
@register_op_impl(aten.to.device)
def non_kwarg_to(fake_mode, func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args, kwargs, normalize_to_only_use_kwargs=True
    )
    input_device = new_kwargs["device"]
    out_device = input_device if input_device else new_kwargs["input"].device
    new_kwargs["device"] = torch.device("meta")
    inp = new_kwargs.pop("input")
    with in_kernel_invocation_manager(fake_mode):
        r = func(inp, **new_kwargs)
    # TODO: I think this does the wrong thing if r is inp
    return fake_mode.fake_tensor_converter.from_meta_and_device(
        fake_mode, r, out_device
    )


def stride_incorrect_op(op):
    return False


# These operators have meta implementations with incorrect strides
@register_op_impl(stride_incorrect_op)
def wordaround_stride_incorrect_op(fake_mode, func, *args, **kwargs):
    # This is a workaround for meta implementations with incorrect strides

    def is_symbolic(x):
        if isinstance(x, FakeTensor):
            return x._has_symbolic_sizes_strides
        if isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return True
        return False

    # For static shapes, we can fall back to eager for the real strides
    if fake_mode.allow_fallback_kernels:
        require_dynamic = any(
            is_symbolic(x) for x in itertools.chain(args, kwargs.values())
        )
        if not require_dynamic:
            flat_args, args_spec = pytree.tree_flatten((args, kwargs))
            return run_fallback_kernel(fake_mode, func, flat_args, args_spec, None)

    raise UnsupportedOperatorException(func)


# Dont default to default device handling,
# since the device of `the_template` is ignored
@register_op_impl(aten.resize_as_.default)
def resize_as_(fake_mode, func, *args, **kwargs):
    with in_kernel_invocation_manager(fake_mode):
        return func(*args, **kwargs)


@register_op_impl(aten._sparse_coo_tensor_with_dims_and_tensors.default)
def _sparse_coo_tensor_with_dims_and_tensors(fake_mode, func, *args, **kwargs):
    # TODO: remove me
    return constructors(fake_mode, func, *args, **kwargs)


# index.Tensor data-dependent in only some conditions
@register_op_impl(
    lambda func: torch.Tag.dynamic_output_shape in func.tags
    and func
    not in [aten.index.Tensor, aten.nonzero.default, aten.repeat_interleave.Tensor]
)
def dyn_shape(fake_mode, func, *args, **kwargs):
    raise DynamicOutputShapeException(func)


def _unique(
    fake_mode,
    func,
    arg,
    dim,
    sorted=True,
    return_inverse=False,
    return_counts=False,
    *,
    unique_consecutive=False,
):
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    nnz = arg.unique_consecutive_memo if unique_consecutive else arg.unique_memo

    # Do not use a memo for unique_dim
    if dim is not None or nnz is None:
        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )

        if not has_free_symbols(arg.numel()) and arg.numel() == 0:
            # If numel is zero, then the output size must be zero.
            # In this case, we must not allocate an unbacked SymInt,
            # because if we do, it will immediately get refined to
            # zero, but this will be inconsistent with size oblivious
            # tests (which will continue to claim that the unbacked
            # symint cannot equal zero).  We could also unconditionally
            # allocate an unbacked SymInt and not refine its range,
            # but this seems more precise.
            nnz = 0
        else:
            nnz = fake_mode.shape_env.create_unbacked_symint()

            maxval = sys.maxsize - 1

            numel = arg.numel() if dim is None else arg.size(dim)
            if not has_free_symbols(numel):
                maxval = int(numel)

            _constrain_range_for_size(nnz, max=maxval)

        if dim is None:
            if unique_consecutive:
                arg.unique_consecutive_memo = nnz
            else:
                arg.unique_memo = nnz

    if dim is None:
        ret = [arg.new_empty((nnz,))]
    else:
        ret = [arg.new_empty(*arg.shape[:dim], nnz, *arg.shape[dim + 1 :])]

    return_if_dim_and_cpu = dim is not None and arg.fake_device == torch.device("cpu")
    if return_inverse or return_if_dim_and_cpu:
        inverse = arg.new_empty(arg.shape if dim is None else (arg.shape[dim],))
    else:
        inverse = arg.new_empty(0)
    ret.append(inverse)

    if return_counts or return_if_dim_and_cpu:
        counts = arg.new_empty(ret[0].shape if dim is None else (ret[0].shape[dim],))
    else:
        counts = arg.new_empty(0)
    ret.append(counts)

    return tuple(ret)


@register_op_impl(aten._unique2.default)
def unique2(
    fake_mode, func, arg, sorted=True, return_inverse=False, return_counts=False
):
    return _unique(fake_mode, func, arg, None, sorted, return_inverse, return_counts)


@register_op_impl(aten.select.int)
def meta_select(fake_mode, func, self, dim, index):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    if self.is_sparse:
        return NotImplemented

    ndim = self.dim()
    torch._check_index(
        ndim != 0,
        lambda: "select() cannot be applied to a 0-dim tensor.",
    )

    dim = dim if dim >= 0 else dim + ndim
    size = self.size(dim)

    new_size = list(self.size())
    new_stride = list(self.stride())

    new_storage_offset = None
    if guard_or_false(index >= 0):
        new_storage_offset = self.storage_offset() + index * new_stride[dim]
    elif guard_or_false(index < 0):
        new_storage_offset = self.storage_offset() + (index + size) * new_stride[dim]

    if new_storage_offset is None:
        if fake_mode.shape_env is None or (
            not fake_mode.shape_env.allow_scalar_outputs
            and not fake_mode.allow_scalar_outputs
        ):
            raise DataDependentOutputException(func)

        # index is data-dependent, we do not know which index we are accessing it could be index or index+size!
        # we assign a new data-dependent symbol for the storage offset.
        new_storage_offset = fake_mode.shape_env.create_unbacked_symint()

    del new_size[dim]
    del new_stride[dim]
    assert new_storage_offset is not None
    return self.as_strided(new_size, new_stride, new_storage_offset)


@register_op_impl(aten.unique_dim.default)
def unique_dim(
    fake_mode, func, arg, dim, sorted=True, return_inverse=False, return_counts=False
):
    return _unique(
        fake_mode,
        func,
        arg,
        # normalize dim to be non-negative
        dim if dim >= 0 else dim % max(arg.ndim, 1),
        sorted,
        return_inverse,
        return_counts,
    )


@register_op_impl(aten.unique_consecutive.default)
def _(fake_mode, func, arg, return_inverse=False, return_counts=False, dim=None):
    return _unique(
        fake_mode,
        func,
        arg,
        dim,
        False,
        return_inverse,
        return_counts,
        unique_consecutive=True,
    )


# This function is python match of computeStride_impl in TensorUtils.cpp
def _compute_stride(old_shape, old_stride, new_shape, size_oblivious=False):
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_eq,
    )

    def maybe_guard_or_false(x):
        if size_oblivious:
            return guard_or_false(x)

        return x

    def maybe_guard_or_true(x):
        if size_oblivious:
            return guard_or_true(x)

        return x

    if len(old_shape) == 0:
        return [1] * len(new_shape)

    numel = reduce(operator.mul, old_shape, 1)
    zero_numel = maybe_guard_or_false(numel == 0)
    if zero_numel and maybe_guard_or_false(sym_eq(old_shape, new_shape)):
        return old_stride

    new_stride = [0] * len(new_shape)

    if zero_numel:
        for view_d in range(len(new_shape) - 1, -1, -1):
            if view_d == len(new_shape) - 1:
                new_stride[view_d] = 1
            else:
                new_stride[view_d] = (
                    max(new_shape[view_d + 1], 1) * new_stride[view_d + 1]
                )
        return new_stride

    view_d = len(new_shape) - 1
    chunk_base_stride = old_stride[-1]
    tensor_numel = 1
    view_numel = 1

    for tensor_d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[tensor_d]

        if tensor_d == 0 or (
            maybe_guard_or_true(old_shape[tensor_d - 1] != 1)
            and maybe_guard_or_true(
                old_stride[tensor_d - 1] != tensor_numel * chunk_base_stride
            )
        ):
            while view_d >= 0 and (
                maybe_guard_or_true(view_numel < tensor_numel)
                or maybe_guard_or_false(new_shape[view_d] == 1)
            ):
                new_stride[view_d] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_d]
                view_d -= 1

            if maybe_guard_or_true(view_numel != tensor_numel):
                return None

            if tensor_d > 0:
                chunk_base_stride = old_stride[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return new_stride


def _view_has_unbacked_input(a, shape):
    from torch.fx.experimental.symbolic_shapes import has_hint

    return (
        any(not has_hint(s) for s in a.size())
        or any(not has_hint(s) for s in a.stride())
        or any(not has_hint(s) for s in shape)
    )


def _view_unbacked_meta(a, shape, size_oblivious_enabled=True):
    from torch._prims import view_of
    from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq

    # Creates a valid shape
    shape = utils.extract_shape_from_varargs(shape, validate=False)

    # Reshape may be given a shape with a -1 length
    # This indicates that the dimension's length should be inferred
    shape = utils.infer_size(shape, a.numel())

    # Special-cases reshaping zero dim tensors
    if a.ndim == 0:
        _a = a
        for length in shape:
            torch._check(length == 1)
            _a = torch._refs.unsqueeze(_a, -1)
        if _a is a:
            return view_of(a)
        else:
            return _a

    # Special-cases reshaping to zero dim tensors
    if len(shape) == 0:
        _a = a
        for length in a.shape:
            torch._check(length == 1)
            _a = torch._refs.squeeze(_a, -1)
        if _a is a:
            return view_of(a)
        else:
            return _a

    shape_numel = reduce(operator.mul, shape, 1)

    torch._check(
        a.numel() == shape_numel,
        lambda: f"Could not reshape a tensor with shape {a.shape} as a tensor with shape {shape}!",
    )

    if len(shape) == len(a.shape) and guard_or_false(sym_eq(shape, a.shape)):
        return view_of(a)

    if is_contiguous_or_false(a) if size_oblivious_enabled else is_contiguous(a):
        strides = make_contiguous_strides_for(shape)
        return a.as_strided(shape, strides)

    new_strides = _compute_stride(
        a.size(), a.stride(), shape, size_oblivious=size_oblivious_enabled
    )

    if new_strides is not None:
        return a.as_strided(shape, new_strides)

    # If we fail to do size oblivious view, and backed_size_oblivious was on,
    # then we redo everything by looking at hints and guarding instead of failing.
    # Also if the expression has unbacked symbols, then we run again with size_oblivious_enabled=False
    # to throw a data dependent error.

    if size_oblivious_enabled and (
        torch.fx.experimental._config.backed_size_oblivious
        or _view_has_unbacked_input(a, shape)
    ):
        return _view_unbacked_meta(a, shape, size_oblivious_enabled=False)

    msg = f"Cannot view a tensor with shape {a.shape} and strides {a.stride()} as a tensor with shape {shape}!"
    raise ValueError(msg)


@register_op_impl(aten.view.default)
@register_op_impl(aten._unsafe_view.default)
def _view_meta(fake_mode, func, a, *shape):
    if torch.fx.experimental._config.backed_size_oblivious or _view_has_unbacked_input(
        a, shape
    ):
        return _view_unbacked_meta(a, shape)
    else:
        return torch._refs._reshape_view_helper(a, *shape, allow_copy=False)


@register_op_impl(aten.view_copy.default)
def _view_meta_copy(fake_mode, func, a, *shape, out=None):
    result = _view_meta(fake_mode, func, a, *shape)
    if out is not None:
        return result

    return pytree.tree_map(
        lambda x: x.clone(memory_format=torch.contiguous_format),
        result,
    )


@register_op_impl(aten.repeat_interleave.Tensor)
def repeat_interleave_tensor(fake_mode, func, repeats, output_size=None):
    if output_size is None:
        if (
            fake_mode.shape_env is None
            or not fake_mode.shape_env.allow_dynamic_output_shape_ops
        ):
            raise DynamicOutputShapeException(func)

        output_size = fake_mode.shape_env.create_unbacked_symint()

        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

        _constrain_range_for_size(output_size)
        # TODO: consider a memo
    return repeats.new_empty(output_size)


@register_op_impl(torch.ops.aten.item.default)
@register_op_impl(torch.ops.aten._local_scalar_dense.default)
def local_scalar_dense(fake_mode, func, arg):
    if (r := arg.item_memo) is not None:
        return r
    if fake_mode.shape_env is None or (
        not fake_mode.shape_env.allow_scalar_outputs
        and not fake_mode.allow_scalar_outputs
    ):
        # Without symints/symfloats, cannot handle this
        raise DataDependentOutputException(func)
    if is_float_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symfloat()
    elif is_integer_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symint()
    elif is_boolean_dtype(arg.dtype):
        r = fake_mode.shape_env.create_unbacked_symbool()
    else:
        raise NotImplementedError(f"local_scalar_dense/item NYI for {arg.dtype}")
    arg.item_memo = r
    return r


@register_op_impl(torch.ops.aten.nonzero_numpy.default)
def nonzero_numpy(fake_mode, func, arg):
    return torch.ops.aten.nonzero.default(arg).unbind(1)


@register_op_impl(torch.ops.aten.nonzero.default)
def nonzero(fake_mode, func, arg):
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    if (nnz := arg.nonzero_memo) is None:
        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )
        from torch.utils._sympy.numbers import IntInfinity
        from torch.utils._sympy.value_ranges import bound_sympy

        if not has_free_symbols(arg.numel()) and arg.numel() == 0:
            # If numel is zero, then the output size must be zero.
            # In this case, we must not allocate an unbacked SymInt,
            # because if we do, it will immediately get refined to
            # zero, but this will be inconsistent with size oblivious
            # tests (which will continue to claim that the unbacked
            # symint cannot equal zero).  We could also unconditionally
            # allocate an unbacked SymInt and not refine its range,
            # but this seems more precise.
            nnz = 0
        else:
            nnz = fake_mode.shape_env.create_unbacked_symint()

            maxval = sys.maxsize - 1

            if not has_free_symbols(arg.numel()):
                maxval = int(arg.numel())
            else:
                prod_node = math.prod(arg.shape).node
                prod_range = bound_sympy(
                    prod_node.expr, prod_node.shape_env.var_to_range
                )
                if isinstance(prod_range.upper, IntInfinity):
                    maxval = sys.maxsize - 1
                else:
                    maxval = prod_range.upper

            _constrain_range_for_size(nnz, max=maxval)

        arg.nonzero_memo = nnz

    return arg.new_empty_strided((nnz, arg.dim()), (1, nnz), dtype=torch.int64)


@register_op_impl(torch.ops.aten._padded_dense_to_jagged_forward.default)
def _padded_dense_to_jagged_forward(fake_mode, func, padded, offsets, total_L=None):
    # only one jagged dim is supported for now
    assert len(offsets) == 1

    if not total_L:
        if (
            fake_mode.shape_env is None
            or not fake_mode.shape_env.allow_dynamic_output_shape_ops
        ):
            # Without symints/symfloats, cannot handle this
            raise DynamicOutputShapeException(func)

        total_L = fake_mode.shape_env.create_unbacked_symint()

        maxval = sys.maxsize - 1

        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import (
            _constrain_range_for_size,
            has_free_symbols,
        )

        if not has_free_symbols(padded.numel()):
            maxval = int(padded.numel())

        _constrain_range_for_size(total_L, min=0, max=maxval)

    output_shape = (total_L, *padded.shape[2:])
    return padded.new_empty(output_shape)


@register_op_impl(torch.ops.aten.masked_select.default)
def masked_select(fake_mode, func, self, mask):
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    nnz = fake_mode.shape_env.create_unbacked_symint()

    # see nonzero for commentary
    maxval = sys.maxsize - 1

    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import (
        _constrain_range_for_size,
        has_free_symbols,
    )
    from torch.utils._sympy.numbers import IntInfinity
    from torch.utils._sympy.value_ranges import bound_sympy

    # If num elements is expressed symbolically, calculate
    # the concrete value based on upper bounds. Otherwise,
    # we can set max val directly.
    if not has_free_symbols(self.numel()):
        num_elements = int(self.numel())
    else:
        prod_node = math.prod(self.shape).node
        prod_range = bound_sympy(prod_node.expr, prod_node.shape_env.var_to_range)
        if isinstance(prod_range.upper, IntInfinity):
            num_elements = sys.maxsize - 1
        else:
            num_elements = prod_range.upper
    if num_elements > 2:
        maxval = num_elements

    _constrain_range_for_size(nnz, max=maxval)

    return self.new_empty((nnz,))


@register_op_impl(torch.ops.aten._assert_tensor_metadata.default)
def assert_tensor_metadata(
    fake_mode,
    func,
    t,
    sizes=None,
    strides=None,
    dtype=None,
    *,
    device=None,
    layout=None,
) -> None:
    if sizes is not None:
        assert t.size() == sizes, (
            f"Tensor sizes mismatch! Expected: {sizes}, Got: {t.size()}"
        )
    if strides is not None:
        assert t.stride() == strides, (
            f"Tensor strides mismatch! Expected: {strides}, Got: {t.stride()}"
        )
    if dtype is not None:
        assert t.dtype == dtype, (
            f"Tensor dtype mismatch! Expected: {dtype}, Got: {t.dtype}"
        )
    if layout is not None:
        assert t.layout == layout, (
            f"Tensor layout mismatch! Expected: {layout}, Got: {t.layout()}"
        )
    if device is not None:
        assert t.device == device, (
            f"Tensor device mismatch! Expected: {device}, Got: {t.device}"
        )


# NB: this must be ordered after local_scalar_dense
@register_op_impl(lambda func: torch.Tag.data_dependent_output in func.tags)
def data_dep(fake_mode, func, *args, **kwargs):
    raise DataDependentOutputException(func)


# Bool Indices get Expanded as Masks
# See: IndexingUtils.h:expandTensors
def check_no_bool_index_tensors(func, self, indices):
    for index in indices:
        if index is not None and index.dtype in (torch.bool, torch.uint8):
            raise DynamicOutputShapeException(func)


def run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    out_device = new_kwargs["input"].device
    with in_kernel_invocation_manager(fake_mode):
        out = func(*args, **kwargs)
        if not is_noncontiguous_supported(out_device):
            out = out.new_empty(out.shape)

    if out is new_kwargs["input"]:
        return out  # copy_
    return FakeTensor(fake_mode, out, out_device)


_is_builtin_namespaces = ordered_set("aten", "prims", "prim")


def is_builtin(op):
    return op.namespace in _is_builtin_namespaces


def has_meta(func):
    return torch._C._dispatch_has_computed_kernel_for_dispatch_key(func.name(), "Meta")


# These are for the `torch._foreach_...` ops like `torch._foreach_add`.
@register_op_impl(
    lambda func: is_builtin(func)
    and func.name().startswith("aten::_foreach_")
    and has_meta(func)
)
def foreach_run_and_map_input_device(fake_mode, func, *args, **kwargs):
    tensor_lists = [
        arg
        for arg in itertools.chain(args, kwargs.values())
        if isinstance(arg, (list, tuple))
        and len(arg)
        and isinstance(arg[0], torch.Tensor)
    ]

    try:
        with in_kernel_invocation_manager(fake_mode):
            out_meta = func(*args, **kwargs)
    except NotImplementedError:
        return NotImplemented

    if not out_meta:
        return out_meta

    assert tensor_lists
    out_fake = []

    for i, meta_t in enumerate(out_meta):
        device, _ = FakeTensor._find_common_device(func, [tl[i] for tl in tensor_lists])
        out_fake.append(
            fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, meta_t, device
            )
        )

    return out_fake


# Dont default to default device handling,
# Since op can take in non-zero sized cpu
# index tensors with cuda self
@register_op_impl(aten.index.Tensor)
def index_tensor(fake_mode, func, *args, **kwargs):
    from torch._meta_registrations import meta_index_Tensor

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    out_device = new_kwargs["input"].device
    # ensure nonzero call goes to fake tensor
    with fake_mode:
        out = meta_index_Tensor(*args, **kwargs)
        return out.to(out_device)


# Can take mixed meta/non-meta arguments; the meta registration
# will roughly do the right thing even when given real devices
@register_op_impl(aten._embedding_bag.default)
def embedding_bag(fake_mode, func, *args, **kwargs):
    from torch._meta_registrations import meta_embedding_bag

    with fake_mode:
        return meta_embedding_bag(*args, **kwargs)


# takes in multiple-devices, dont default to default device handling
@register_op_impl(aten._unsafe_index_put.default)
@register_op_impl(aten.copy.default)
@register_op_impl(aten.copy_.default)
@register_op_impl(aten.slice_scatter.default)
def multi_device_op_default(fake_mode, func, *args, **kwargs):
    return run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)


# same with multi_device_op_default, but return the input
@register_op_impl(aten.copy.out)
@register_op_impl(aten.slice_scatter.out)
def multi_device_op_out(fake_mode, func, *args, **kwargs):
    with in_kernel_invocation_manager(fake_mode):
        func(*args, **kwargs)

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    return new_kwargs["input"]


@register_op_impl(aten.index_put.default)
@register_op_impl(aten.index_put_.default)
def index_put_impl(fake_mode, func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    values = new_kwargs["values"]
    self_device = new_kwargs["input"].fake_device
    torch._check(
        self_device == values.fake_device or (values.ndim == 0 and values.numel() == 1),
        lambda: f"Mismatching {func} device between self ({self_device}) and values ({values.device})",
    )

    out = run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)
    if func is aten.index_put_.default:
        return new_kwargs["input"]
    else:
        return out


@register_op_impl(aten._nested_tensor_from_tensor_list.default)
@register_op_impl(aten._nested_tensor_from_tensor_list.out)
@register_op_impl(aten._nested_view_from_buffer.default)
@register_op_impl(aten._nested_view_from_buffer_copy.default)
def nested_tensors_unsupported(fake_mode, func, *args, **kwargs):
    raise UnsupportedOperatorException(
        "torch.compile does not support strided NestedTensor"
    )


@register_op_impl(
    [
        x
        for x in _device_not_kwarg_ops
        if x
        not in (
            # these are already registered elsewhere
            aten.is_pinned.default,
            aten.to.device,
            aten.to.prim_Device,
            aten._nested_tensor_from_tensor_list.default,
            aten._nested_tensor_from_tensor_list.out,
        )
    ]
)
def nyi(fake_mode, func, *args, **kwargs):
    assert func not in _device_not_kwarg_ops, f"NYI: {func}"


@register_op_impl([aten.convolution.default, aten.convolution_backward.default])
def conv(fake_mode, func, *args, **kwargs):
    _, kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    device = kwargs["input"].fake_device
    # need to re-enable mode so the tensors report fake device
    with fake_mode:
        # if the input is unsqueezed is done in Convolution.cpp we get segfault
        k = kwargs["weight"].ndim
        batch = kwargs["input"].shape[0]

        # Avoid importing sympy at a module level
        from torch.fx.experimental.symbolic_shapes import has_hint

        if not has_hint(batch):
            # TODO: We can make this a little more faithful with best effort
            # channels last detection (but only if it's statically obvious!)
            mem_fmt = None
        elif k == 3 and not kwargs["input"].is_mkldnn and not kwargs["input"].is_xpu:
            mem_fmt = None
        else:
            if func is aten.convolution.default:
                conv_backend = torch._C._select_conv_backend(**kwargs)
            else:
                conv_backend = torch._C._select_conv_backend(
                    kwargs["input"],
                    kwargs["weight"],
                    bias=None,
                    stride=kwargs["stride"],
                    padding=kwargs["padding"],
                    dilation=kwargs["dilation"],
                    transposed=kwargs["transposed"],
                    output_padding=kwargs["output_padding"],
                    groups=kwargs["groups"],
                    bias_sizes=kwargs["bias_sizes"],
                )
            mem_fmt = torch._C._conv_determine_backend_memory_format(
                kwargs["input"], kwargs["weight"], conv_backend
            )

    def convert(t, mem_fmt):
        if t is None:
            return t
        if mem_fmt is not None:
            t = t.to(memory_format=mem_fmt)
        return FakeTensor(fake_mode, t, device)

    with in_kernel_invocation_manager(fake_mode):
        out = func(**kwargs)

        if func is aten.convolution.default:
            return convert(out, mem_fmt)
        else:
            return (
                convert(out[0], mem_fmt),
                convert(out[1], mem_fmt),
                convert(out[2], None),
            )


@register_op_impl(torch.ops.aten.bincount.default)
def bincount(fake_mode, func, inputs, weights=None, minlength=0):
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    new_size = fake_mode.shape_env.create_unbacked_symint()

    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    _constrain_range_for_size(new_size)
    torch._check(new_size >= minlength)
    return inputs.new_empty(new_size)


@register_op_impl(torch.ops.aten._pack_padded_sequence.default)
def _pack_padded_sequence(fake_mode, func, inputs, lengths, batch_first):
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    new_batch_size = fake_mode.shape_env.create_unbacked_symint()

    from torch.fx.experimental.symbolic_shapes import _constrain_range_for_size

    _constrain_range_for_size(new_batch_size)

    if not batch_first:
        # Inputs should have shape (batch_size, seq_len, *)
        inputs = inputs.transpose(0, 1)

    res_size = inputs.shape[1:]
    packed_data = inputs.new_empty(res_size)
    batch_size = inputs.new_empty((new_batch_size,))
    return (packed_data, batch_size)


FAST_OP_IMPLEMENTATIONS = {}


# Unlike register_op_impl, these don't do the slow iteration for
# run_impl_check, and these run BEFORE decompositions
def register_fast_op_impl(func: OpOverload):
    def impl_decorator(op_impl):
        FAST_OP_IMPLEMENTATIONS[func] = op_impl
        return op_impl

    return impl_decorator


# infer_size_impl in ExpandUtils
def infer_size(a, b):
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        torch._check(
            guard_or_false(sizeA == 1) or guard_or_false(sizeB == 1) or sizeA == sizeB,
            lambda: f"The size of tensor a ({sizeA}) "
            f"must match the size of tensor b ({sizeB}) "
            f"at non-singleton dimension {i})",
        )
        expandedSizes[i] = sizeB if guard_or_false(sizeA == 1) else sizeA
    return tuple(expandedSizes)


def make_fast_binary_impl(
    slow_ref, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
):
    def fast_binary_impl(mode, *args, **kwargs):
        def slow(msg):
            count_label(f"slow {msg}")
            with mode:
                return slow_ref(*args, **kwargs)

        count_label("attempt fast")

        # Fast path (based off of TensorIterator fast path).
        # Unfortunately, there is no way to easily deduplicate
        # this with either the TensorIterator C++ implementation
        # (which we don't want to SymIntify, and also the algorithm
        # here is slightly different from TensorIterator to allow
        # for broadcasting), nor the PrimTorch implementation
        # (which does not actually implement a fast path.)

        operands = args

        # compute_shape
        final_shape = None
        for op in operands:
            shape = op.shape if isinstance(op, torch.Tensor) else ()
            if final_shape is None:
                final_shape = shape
            # TODO: Minor optimization: track if the shapes
            # were equal so you can skip the equality check
            # below if unnecessary
            final_shape = infer_size(final_shape, shape)
        assert final_shape is not None

        from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq

        # Do some extra safety checks to see if the output
        # stride is obvious
        for op in operands:
            if (
                isinstance(op, torch.Tensor)
                and len(op.shape) == len(final_shape)
                # take the slow path if result is not determined.
                and guard_or_false(sym_eq(op.shape, final_shape))
            ):
                break
        else:
            # if we never break in the for loop above we take the slow path.
            return slow("both tensors nontrivially broadcast")

        # compute_types
        cpu = torch.device("cpu")
        common_device = cpu
        common_dtype = None
        has_different_input_dtypes = False
        for op in operands:
            if not isinstance(op, torch.Tensor):
                # Use elementwise_dtypes for the tricky case
                has_different_input_dtypes = True
                continue
            if common_device == cpu and not op.device.type == "cpu":
                common_device = op.device
            # Slightly simplified here as target_dtype cannot vary
            if common_dtype is None:
                common_dtype = op.dtype
            elif common_dtype != op.dtype:
                has_different_input_dtypes = True

        if has_different_input_dtypes:
            # compute promotion
            # TODO: we don't need the compute type
            _, common_dtype = elementwise_dtypes(
                *operands, type_promotion_kind=type_promotion_kind
            )

        # check all tensors on same device
        # cpu scalars are assumed allow
        current_cpu_scalars_on_non_cpu = 0
        max_cpu_scalars_on_non_cpu = 1  # hard coded atm
        for op in operands:
            if not isinstance(op, torch.Tensor):
                continue
            if common_device != cpu and op.dim() == 0 and op.device == cpu:
                if current_cpu_scalars_on_non_cpu >= max_cpu_scalars_on_non_cpu:
                    return slow("error")
                current_cpu_scalars_on_non_cpu += 1
            elif op.device != common_device:
                return slow("error")

        # compute_fast_setup_type
        definitely_contiguous = True
        definitely_channels_last = True

        # TODO: is_non-overlapping_and_dense not bound from Python
        # no inplace, no out, everything defined

        if is_noncontiguous_supported(common_device):
            for op in operands:
                if not isinstance(op, torch.Tensor):
                    continue
                definitely_contiguous = (
                    definitely_contiguous
                    and contiguous_for_memory_format_or_false(
                        op, memory_format=torch.contiguous_format
                    )
                )
                definitely_channels_last = (
                    definitely_channels_last
                    and contiguous_for_memory_format_or_false(
                        op, memory_format=torch.channels_last
                    )
                )
        if definitely_contiguous:
            # do contiguous
            count_label("fast is_contiguous")
            return FakeTensor(
                mode,
                torch.empty(
                    final_shape,
                    dtype=common_dtype,
                    device="meta",
                    memory_format=torch.contiguous_format,
                ),
                device=common_device,
            )
        if definitely_channels_last:
            count_label("fast channels_last")
            # do channels last
            return FakeTensor(
                mode,
                torch.empty(
                    final_shape,
                    dtype=common_dtype,
                    device="meta",
                    memory_format=torch.channels_last,
                ),
                device=common_device,
            )

        return slow("no contiguity match")

    return fast_binary_impl


# disable the python dispatcher to avoid decomposing detach() further
# (proxy_mode should still decompose detach() though)
def fast_detach(fake_mode, x, include_real=False):
    with no_python_dispatcher(), in_kernel_invocation_manager(fake_mode):
        out = torch.ops.aten.detach.default(x)
    if include_real:
        return FakeTensor(fake_mode, out, x.device, real_tensor=x.real_tensor)
    return FakeTensor(fake_mode, out, x.device)


@functools.cache
def get_fast_op_impls():
    import torch._refs

    register_fast_op_impl(torch.ops.aten.add.Tensor)(
        make_fast_binary_impl(torch._refs.add)
    )
    register_fast_op_impl(torch.ops.aten.sub.Tensor)(
        make_fast_binary_impl(torch._refs.sub)
    )
    register_fast_op_impl(torch.ops.aten.mul.Tensor)(
        make_fast_binary_impl(torch._refs.mul)
    )  # type: ignore[has-type]
    register_fast_op_impl(torch.ops.aten.div.Tensor)(
        make_fast_binary_impl(
            torch._refs.div,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        )
    )
    register_fast_op_impl(torch.ops.aten.detach.default)(fast_detach)
    return FAST_OP_IMPLEMENTATIONS

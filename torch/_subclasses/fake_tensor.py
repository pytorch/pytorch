import contextlib
import functools
import itertools
import logging
import os
import traceback
import weakref
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType

import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_boolean_dtype,
    is_float_dtype,
    is_integer_dtype,
)
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.experimental.symbolic_shapes import (
    DimConstraint,
    DimDynamic,
    free_symbols,
)
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)

from torch.utils._pytree import PyTree, tree_flatten, tree_map, tree_map_only
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef

DimList = List

log = logging.getLogger(__name__)
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

pytree = torch.utils._pytree
T = TypeVar("T")
TensorWeakRef = Any

aten = torch._ops.ops.aten

CONSTANT_NUMEL_LIMIT = 1

RECURSION_COUNT = 0


# Small helper that increments recursion count, and
# resets it when the object goes out of scope.  Useful
# if you don't want to increase indentation which is
# what a context manager would do.
class IncrementRecursionCount:
    def __init__(self):
        global RECURSION_COUNT
        RECURSION_COUNT += 1

    def __del__(self):
        global RECURSION_COUNT
        RECURSION_COUNT -= 1


@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    reason: str


@dataclass
class DynamicOutputShapeException(RuntimeError):
    func: OpOverload


@dataclass
class DataDependentOutputException(RuntimeError):
    func: OpOverload


@dataclass
class UnsupportedOperatorException(RuntimeError):
    func: OpOverload


_device_not_kwarg_ops = (
    aten._resize_output_.default,
    aten._nested_tensor_from_tensor_list.default,
    aten._nested_tensor_from_tensor_list.out,
    aten.pin_memory.default,
    aten.is_pinned.default,
    aten.to.device,
    aten.to.prim_Device,
    aten._pin_memory.default,
    aten._pin_memory.out,
    aten._resize_output.default,
    aten._resize_output.out,
)

# this op is never actually used
_non_kwarg_device_constructors = (aten._list_to_tensor,)


# This function indicates if the backend device
# supports non-contiguous tensors
def is_noncontiguous_supported(device):
    if device.type == "hpu":
        return False
    return True


def contains_tensor_types(type):
    tensor_type = torch._C.TensorType.get()
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )


_like_tensor_constructors = (
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


@functools.lru_cache(None)
def _is_tensor_constructor(func: OpOverload):
    assert isinstance(func, OpOverload)
    schema = func._schema
    if any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return False
    # TODO: no real reason to restrict multiple outputs
    return (
        len(schema.returns) == 1 and schema.returns[0].type is torch._C.TensorType.get()
    )


def is_fake(x):
    if isinstance(x, FakeTensor):
        return True
    if is_traceable_wrapper_subclass(x):
        flattened_tensors, _ = type(x).__tensor_flatten__(x)
        # need to recurse because we could have nested subclasses
        all_fake = all(is_fake(x) for x in flattened_tensors)
        any_fake = any(is_fake(x) for x in flattened_tensors)
        assert all_fake == any_fake, "got mixed fake and real tensors!"
        return all_fake
    return False


@functools.lru_cache(None)
def get_schema_info(func):
    return torch._C._SchemaInfo(func._schema)  # type: ignore[attr-defined]


# many of the decompositions registered to torch/_prims do not at the moment model
# aliasing or strides, so as an incremental step, just enable the decompositions in
# torch/_decomp/decompositions.py.
# decomps are used for aot autograd tracing so we would like to unify on their
# implementation and add additional testing to them
@functools.lru_cache(None)
def torch_decomp_decompositions(func):
    from torch._decomp import decomposition_table

    decompositions = torch._decomp.decompositions
    decomp_attrs = [getattr(decompositions, attr) for attr in dir(decompositions)]
    return decomposition_table[func] in decomp_attrs


def tree_flatten_only(ty: Type[T], pytree: PyTree):
    flat_vals, _ = tree_flatten(pytree)
    return [elem for elem in flat_vals if isinstance(elem, ty)]


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakIdRef` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverter:
    @property
    def tensor_memo(self):
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter
    constant_storage_mapping: Dict[StorageWeakRef, List[ReferenceType]]

    def __init__(self):
        self.meta_converter = MetaConverter()

        # map from to storage to corresponding constant tensors
        self.constant_storage_mapping = {}

    def add_constant_storage_mapping(self, fake_tensor):
        # when you have a constant, aliased tensor:
        # const_tensor.add_(torch.rand([1]))
        # all aliases of it must become no longer const
        assert isinstance(fake_tensor, FakeTensor) and fake_tensor.constant is not None
        weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())

        # we need a map from a weak storage to all of its corresponding
        # constant tensors. python doesn't have the weak value equivalent
        # of defaultdict(list), so we are using a WeakValueDictionary as one
        if weak_st not in self.constant_storage_mapping:
            self.constant_storage_mapping[weak_st] = []
        self.constant_storage_mapping[weak_st].append(weakref.ref(fake_tensor))

    def invalidate_constant_aliases(self, tensor):
        assert not isinstance(tensor, FakeTensor)

        weak_st = StorageWeakRef(tensor._typed_storage())
        if weak_st not in self.constant_storage_mapping:
            return

        for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
            ten = weak_tensor_ref()
            if ten is not None:
                ten._fix_weakref()
                ten.constant = None

        del self.constant_storage_mapping[weak_st]

    def _get_memo(self, t):
        if WeakIdRef(t) in self.tensor_memo:
            out = self.tensor_memo[WeakIdRef(t)]
            out._fix_weakref()
            return out
        return None

    def set_tensor_memo(self, t, v):
        th = WeakIdRef(t)

        # hold a weak ref to self, otherwise it will be kept alive
        # by the del_ten closure
        self_weak_ref = weakref.ref(self)

        def del_ten():
            self_ref = self_weak_ref()
            if self_ref is None:
                return
            # on shutdown, th may not be in memo
            self_ref.tensor_memo.pop(th, None)

        weakref.finalize(t, del_ten)
        self.tensor_memo[th] = v

    def from_real_tensor(
        self,
        fake_mode,
        t,
        make_constant=False,
        shape_env=None,
        ignore_subclass=False,
        *,
        source=None,
        dynamic_dims: Optional[DimList[DimDynamic]] = None,
        constraint_dims: Optional[DimList[DimConstraint]] = None,
        memoized_only=False,
    ):
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        if memoized_only:
            return None
        existing_device = t.device
        # not yet supported in metatensors
        if t.is_quantized:
            raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        if type(t) is torch.nn.Parameter:
            assert not make_constant

        def mk_fake_tensor(make_meta_t):
            # NB: don't use in_kernel_invocation_manager. to
            # ensure FakeTensor can internally do constant computation
            # as necessary.  Invocation manager is "more correct" as
            # it works for more operators in make_meta_t, but
            # invariant is that make_meta_t only calls factories
            # for which it is not strictly necessary to use the
            # invocation manager (I think!)
            with no_dispatch():
                return FakeTensor(
                    fake_mode,
                    make_meta_t(),
                    existing_device,
                    constant=t if make_constant else None,
                )

        out = self.meta_converter(
            t,
            shape_env=shape_env,
            callback=mk_fake_tensor,
            ignore_subclass=ignore_subclass,
            source=source,
            dynamic_dims=dynamic_dims,
            constraint_dims=constraint_dims,
        )
        if out is NotImplemented:
            raise UnsupportedFakeTensorException("meta converter nyi")
        if make_constant:
            self.add_constant_storage_mapping(out)
        # NB: meta_converter set the memo
        return out

    # If you specify the device, it MUST be a meta tensor.
    def from_meta_and_device(self, fake_mode, t, device):
        assert (
            t.device.type == "meta"
        ), f"tensor's device must be `meta`, got {t.device.type} instead"
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensor(fake_mode, t, device)
        self.set_tensor_memo(t, out)
        return out

    # You can have a real tensor that you need to convert into a fake tensor.
    # If you have a meta tensor already, call from_meta_and_device.
    #
    # You're allowed to pass a meta tensor to be turned into a fake
    # tensor; although an odd thing to do, this can occur if you're doing
    # cross ref testing and the inner test is already operating on meta tensors.
    def __call__(
        self,
        fake_mode,
        t,
        *,
        make_constant=False,
        shape_env=None,
        ignore_subclass=False,
        source=None,
        dynamic_dims=None,
        constraint_dims=None,
        memoized_only=False,
    ):
        return self.from_real_tensor(
            fake_mode,
            t,
            make_constant,
            shape_env=shape_env,
            ignore_subclass=ignore_subclass,
            source=source,
            dynamic_dims=dynamic_dims,
            constraint_dims=constraint_dims,
            memoized_only=memoized_only,
        )


op_implementations = []


def register_op_impl(run_impl_check: Union[Callable[[OpOverload], bool], OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations
        if isinstance(run_impl_check, OpOverload):
            op_implementations.append((lambda func: func == run_impl_check, op_impl))
        else:
            op_implementations.append((run_impl_check, op_impl))

        return op_impl

    return impl_decorator


@register_op_impl(
    lambda func: (_is_tensor_constructor(func) or func in _like_tensor_constructors)
)
def constructors(fake_mode, func, *args, **kwargs):
    assert func not in _non_kwarg_device_constructors
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
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


@register_op_impl(lambda func: func in (aten.to.prim_Device, aten.to.device))
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
    if op.namespace not in ("aten", "prims"):
        return False
    if op is aten._fft_c2c.default:
        return False

    op_name = op.name()
    if "fft" in op_name:
        return True
    return False


# These operators have meta implementations with incorrect strides
@register_op_impl(stride_incorrect_op)
def wordaround_stride_incorrect_op(fake_mode, func, *args, **kwargs):
    # This is a workaround for meta implmentations with incorrect strides

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
            return run_fallback_kernel(fake_mode, func, args, kwargs, None)

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


@register_op_impl(lambda func: func is aten.repeat_interleave.Tensor)
def repeat_interleave_tensor(fake_mode, func, repeats, output_size=None):
    if output_size is None:
        raise DynamicOutputShapeException(func)
    return repeats.new_empty(output_size)


@register_op_impl(lambda func: func is torch.ops.aten._local_scalar_dense.default)
def local_scalar_dense(fake_mode, func, arg):
    if fake_mode.shape_env is None or not fake_mode.shape_env.allow_scalar_outputs:
        # Without symints/symfloats, cannot handle this
        raise DataDependentOutputException(func)
    if is_float_dtype(arg.dtype):
        return fake_mode.shape_env.create_unbacked_symfloat()
    elif is_integer_dtype(arg.dtype):
        return fake_mode.shape_env.create_unbacked_symint()
    elif is_boolean_dtype(arg.dtype):
        return fake_mode.shape_env.create_unbacked_symbool()
    else:
        raise NotImplementedError(f"local_scalar_dense/item NYI for {arg.dtype}")


@register_op_impl(lambda func: func is torch.ops.aten.nonzero.default)
def nonzero(fake_mode, func, arg):
    if (
        fake_mode.shape_env is None
        or not fake_mode.shape_env.allow_dynamic_output_shape_ops
    ):
        # Without symints/symfloats, cannot handle this
        raise DynamicOutputShapeException(func)

    if arg.nonzero_memo is None:
        import sys

        from torch.fx.experimental.symbolic_shapes import constrain_range

        nnz = fake_mode.shape_env.create_unbacked_symint()

        # This is unsound, but it works well in practice
        # See https://docs.google.com/document/d/1lFRYAJo5nrfxRhwIzGnfi2pbLpU6T4ytSRSuLJ5qebI/edit#
        # TODO: Add a config knob to turn off this unsound behavior
        #
        # NB: If numel < 2, the bounds here might be COMPLETELY
        # disjoint with what can actually occur.  But this is fine:
        # remember, the hypothesis is that if your later code works
        # with N >= 2, it will work with N = 1 and N = 0.
        maxval = sys.maxsize - 1
        if not free_symbols(arg.numel()):
            # Don't upgrade the range if numel is less than two, since we then
            # have an empty range which makes things go explodey.  We also
            # don't allow for 2 because that would specialize the unbacked
            # SymInt to 2, which is also likely to be buggy.
            if arg.numel() >= 2:
                maxval = arg.numel()
        constrain_range(nnz, min=2, max=maxval)

        arg._nonzero_memo = nnz
        arg._nonzero_memo_vc = arg._version

    return arg.new_empty((arg.nonzero_memo, arg.dim()), dtype=torch.int64)


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
@register_op_impl(aten.index_put.default)
@register_op_impl(aten._unsafe_index_put.default)
@register_op_impl(aten.copy.default)
@register_op_impl(aten.copy_.default)
@register_op_impl(aten.slice_scatter.default)
def multi_device_op_default(fake_mode, func, *args, **kwargs):
    return run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)


# same with multi_device_op_default, but return the input
@register_op_impl(aten.index_put_.default)
@register_op_impl(aten.copy.out)
@register_op_impl(aten.slice_scatter.out)
def multi_device_op_out(fake_mode, func, *args, **kwargs):
    with in_kernel_invocation_manager(fake_mode):
        out = func(*args, **kwargs)

    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    return new_kwargs["input"]


@register_op_impl(lambda fn: fn in _device_not_kwarg_ops)
def nyi(fake_mode, func, *args, **kwargs):
    assert func not in _device_not_kwarg_ops, f"NYI: {func}"


@register_op_impl(
    lambda func: func in (aten.convolution.default, aten.convolution_backward.default)
)
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
        if not (sizeA == sizeB or sizeA == 1 or sizeB == 1):
            raise RuntimeError(
                f"The size of tensor a ({sizeA}) "
                f"must match the size of tensor b ({sizeB}) "
                f"at non-singleton dimension {i})"
            )
        expandedSizes[i] = sizeB if sizeA == 1 else sizeA
    return tuple(expandedSizes)


def make_fast_binary_impl(slow_ref):
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
        has_scalars = False
        has_tensors = False
        final_shape = None
        for op in operands:
            shape = op.shape if isinstance(op, torch.Tensor) else ()
            if len(shape) == 0:
                has_scalars = True
            else:
                has_tensors = True
            if final_shape is None:
                final_shape = shape
            # TODO: Minor optimization: track if the shapes
            # were equal so you can skip the equality check
            # below if unnecessary
            final_shape = infer_size(final_shape, shape)
        assert final_shape is not None

        # Do some extra safety checks to see if the output
        # stride is obvious
        for op in operands:
            if isinstance(op, torch.Tensor) and op.shape == final_shape:
                break
        else:
            return slow("both tensors nontrivially broadcast")

        # compute_types
        cpu = torch.device("cpu")
        common_device = cpu
        common_dtype = None
        output_dtype = None
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
                *operands, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
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
        is_contiguous = True
        is_channels_last = True
        # TODO: is_non-overlapping_and_dense (not bound from Python
        # no inplace, no out, everything defined

        if is_noncontiguous_supported(common_device):
            for op in operands:
                if not isinstance(op, torch.Tensor):
                    continue
                is_contiguous = is_contiguous and op.is_contiguous(
                    memory_format=torch.contiguous_format
                )
                is_channels_last = is_channels_last and op.is_contiguous(
                    memory_format=torch.channels_last
                )
        if is_contiguous:
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
        if is_channels_last:
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


@functools.lru_cache(None)
def get_fast_op_impls():
    import torch._refs

    register_fast_op_impl(torch.ops.aten.add.Tensor)(
        make_fast_binary_impl(torch._refs.add)
    )
    register_fast_op_impl(torch.ops.aten.sub.Tensor)(
        make_fast_binary_impl(torch._refs.sub)
    )
    register_fast_op_impl(torch.ops.aten.mul.Tensor)(make_fast_binary_impl(torch._refs.mul))  # type: ignore[has-type]
    register_fast_op_impl(torch.ops.aten.div.Tensor)(
        make_fast_binary_impl(torch._refs.div)
    )
    return FAST_OP_IMPLEMENTATIONS


@functools.lru_cache(None)
def init_cuda_context():
    # Backward will error with cuda Fake Tensors if no cuda tensors have been initialized first
    if torch.cuda.is_available():
        torch.empty(1, device="cuda") if torch.version.hip is None else torch.zeros(
            1, device="cuda"
        )


@contextlib.contextmanager
def in_kernel_invocation_manager(fake_mode):
    # See: note [Fake Tensor Dispatch Keys]
    prev_in_kernel = fake_mode.in_kernel_invocation
    meta_in_tls = torch._C._meta_in_tls_dispatch_include()
    assert meta_in_tls == prev_in_kernel, f"{meta_in_tls}, {prev_in_kernel}"

    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    fake_mode.in_kernel_invocation = True
    torch._C._set_meta_in_tls_dispatch_include(True)
    try:
        yield
    finally:
        fake_mode.in_kernel_invocation = prev_in_kernel
        torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel)
        del guard


# Return if the function allows Python numbers to bind to Tensors
def should_allow_numbers_as_tensors(func: OpOverload):
    return torch._C._should_allow_numbers_as_tensors(
        func.name().split("::")[-1].split(".")[0]
    )


class FakeTensorConfig:
    debug = os.environ.get("TORCH_FAKE_TENSOR_DEBUG", False)


class FakeTensor(torch.Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """

    fake_device: torch.device
    fake_mode: "FakeTensorMode"
    constant: Optional[torch.Tensor]

    # This memorizes the unbacked SymInt representing the number of nonzero
    # elements in this tensor.  This is helpful if you do something like
    # x[mask] and y[mask]; mask.nonzero() gets repeatedly called and should
    # give a consistent unbacked SymInt.  It needs to be invalidated in the
    # same way constant is.
    # TODO: Generalize this as needed, e.g., into a trie of memos
    _nonzero_memo: Optional[torch.SymInt]
    _nonzero_memo_vc: Optional[int]

    @property
    def nonzero_memo(self):
        if self._nonzero_memo is None:
            return None
        # Version counter based tracking isn't 100% sound but it's close
        # enough
        if self._nonzero_memo_vc != self._version:
            self._nonzero_memo = None
            return None
        return self._nonzero_memo

    @property
    def device(self):
        if self.fake_mode.in_kernel_invocation:
            return torch.device("meta")
        else:
            return self.fake_device

    # Note: [Fake Tensor Dispatch Keys]
    # In order to model the behavior of device-specific autocast
    # and autograd logic, we update the dispatch keys of FakeTensors
    # to reflect their fake device. This includes the BackendComponent
    # (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
    # related Autocast and Autograd keys. __torch__dispatch__ sits below
    # Autocast and Autograd, and is only invoked when we are at the
    # kernel for the BackendComponent. Then, we add Meta to the
    # thread-local dispatch include set to hit the meta kernel
    # instead of the kernel of the BackendComponent for the fake device.
    # The `device_for_backend_keys` does that below

    @staticmethod
    def __new__(cls, fake_mode, elem, device, constant=None):
        self = torch.Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )

        assert elem.device.type == "meta", elem.device.type
        device = device if isinstance(device, torch.device) else torch.device(device)
        # NB: it is fine, if a little confusing, for device to be meta
        # (we are faking a meta tensor in that case).  However, it often
        # indicates some sort of confusion (e.g., you accidentally passed
        # in a meta tensor when you should have passed in the real tensor).
        # So by default we disallow meta, and if you are working in a situation
        # where it is helpful (e.g., crossref testing) you can turn it back
        # on
        if not fake_mode.allow_meta:
            assert device.type != "meta"
        # normalize device.
        if device.type == "cuda":
            init_cuda_context()

        if (
            device.type in ["cuda", "hpu", torch._C._get_privateuse1_backend_name()]
            and device.index is None
        ):
            device = torch.device(
                f"{device.type}:{getattr(torch, device.type).current_device()}"
            )
        self.fake_device = device  # type: ignore[attr-defined]
        self.fake_mode = fake_mode  # type: ignore[attr-defined]
        self.constant = constant  # type: ignore[attr-defined]
        self._nonzero_memo = None  # type: ignore[attr-defined]
        self._nonzero_memo_vc = None  # type: ignore[attr-defined]

        if FakeTensorConfig.debug:
            import traceback

            self._debug_trace = traceback.extract_stack()  # type: ignore[attr-defined]
        return self

    # In some circumstances, a conventional torch.Tensor constructor
    # will get rewritten to call into FakeTensor.  We must provide an
    # __init__ method that can accept the Python interpreters initialization
    # in such a situation; we must also be able to handle direct fake
    # tensor construction via FakeTensor().
    #
    # In particular, the __init__ call will look funny in the following case:
    #
    #   with FakeTensorMode():
    #       x = torch.Tensor([1, 2, 3])
    #
    # this desugars into:
    #
    #   with FakeTensorMode():
    #       x = torch.Tensor.__new__([1, 2, 3])
    #       # NB: x is a fake tensor, because of the mode!
    #       x.__init__([1, 2, 3])  # not the normal fake tensor args!
    #
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def from_tensor(t, fake_mode):
        return fake_mode.from_tensor(t)

    @classmethod
    @count
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # need to handle here to avoid infinite recursion
        # see [in_kernel_invocation]
        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

        # Because fake mode can return NotImplemented (if it sees a subclass
        # it doesn't know how to deal with), this test here is important
        # because the next dispatch after a fake mode will attempt to use
        # subclasses of tensors to dispatch, and any FakeTensor arguments
        # will be considered eligible.
        unrecognized_types = [
            t for t in types if not issubclass(t, FakeTensor) and t is not torch.Tensor
        ]
        if unrecognized_types:
            not_implemented_log.debug(
                "FakeTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        fake_mode = None
        for arg in itertools.chain(tree_flatten(args)[0], tree_flatten(kwargs)[0]):
            if isinstance(arg, FakeTensor):
                fake_mode = arg.fake_mode
                break

        assert fake_mode is not None

        # If the fake mode is already active, don't try to reapply it!
        # NotImplemented is the right thing to return here, because the
        # typical situation this can occur is if ProxyTensorMode returned a
        # NotImplemented because of a not implemented subclass; we may have
        # unluckily attempted to hit FakeTensor's dispatch first,
        # NotImplemented lets us keep chaining until we find the actual
        # subclass
        cur_stack = _get_current_dispatch_mode_stack()
        # NB: This must test for ANY fake tensor mode, because we want the
        # fake tensor mode to take precedence
        fake_modes_on_stack = [m for m in cur_stack if isinstance(m, FakeTensorMode)]
        if fake_modes_on_stack:
            not_implemented_log.debug(
                "FakeTensor mode already active: %s in %s",
                fake_modes_on_stack,
                cur_stack,
            )
            return NotImplemented

        with fake_mode:  # type: ignore[attr-defined]
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(func, args, kwargs) -> Tuple[torch.device, bool]:
        # Returns: (common_device, has_scalar_only_inputs)

        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite the common_device if it the only existing
        # device comes from a cpu zero-dim tensor
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        def cpu_zero_dim(t):
            return t.device.type == "cpu" and t.dim() == 0

        def merge_devices(t):
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            if not isinstance(t, FakeTensor):
                return

            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            t_is_cpu_zero_dim = cpu_zero_dim(t)
            if t.device == common_device:
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # mismatching devices !
            # if current tensor is cpu 0 dim, defer to existing device
            if t_is_cpu_zero_dim:
                return

            # current device is from cpu 0 dim tensor, overwrite
            if is_cpu_zero_dim:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # mismatching devices of non-zero dim tensors, throw
            # This might be valid behavior and need to be explicitly modeled, e.g. reshape_as
            raise RuntimeError(
                f"Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}"
            )

        tree_map(merge_devices, args)
        tree_map(merge_devices, kwargs)

        # some functions that allow Python numbers to bind to Tensors
        # if we have failed to find a device, and we're running one of these operators,
        # we must have scalar only inputs
        if should_allow_numbers_as_tensors(func) and common_device is None:
            # ops with scalar only inputs always have result on cpu
            has_scalar_only_inputs = True
            common_device = torch.device("cpu")

        return common_device, has_scalar_only_inputs

    __torch_function__ = torch._C._disabled_torch_function_impl


# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with FakeTensorMode()`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will no be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly increase.


class FakeTensorMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        allow_fallback_kernels=True,
        allow_non_fake_inputs=False,
        shape_env=None,
    ):
        log.debug("create_mode 0x%x", id(self))
        self.allow_fallback_kernels = allow_fallback_kernels
        self.fake_tensor_converter = FakeTensorConverter()

        import torch._functorch.config

        self.allow_meta = torch._functorch.config.fake_tensor_allow_meta

        # A flag that controls, whether we want to invoke ops on mix of
        # real weights/global variables and fake inputs
        self.allow_non_fake_inputs = allow_non_fake_inputs

        # [in_kernel_invocation]
        # when FakeTensor is invoked in user code, .device should return
        # the fake_device of the tensor so that code such as as `if x.is_cuda`
        # or torch.zeros([10, 10], device=x.device) continues to execute as if
        # the FakeTensor were real. However, within kernel execution, we return
        # the `Meta` device because all computation within the kernels should
        # behave as if the Tensors are on meta devices. Kernels should allocate
        # new tensors on meta devices, and checks like `is_meta` should return true.
        # within python refs, we always return the real device by defining
        # the device property
        self.in_kernel_invocation = False

        # True if we enter'ed and actually enabled fake tensor mode,
        # false if it was a no-op.  Not thread safe but neither is
        # in_kernel_invocation
        self.enter_stack: List[bool] = []

        self.shape_env = shape_env

        self.stack = "".join(traceback.format_stack())

    # Typically, there is only one fake tensor mode and you test for it by
    # doing an isinstance test.  However, in some situations, there might be
    # TWO fake tensor modes.  The canonical example of this is exporting
    # a fake model: there is an outer fake mode created by the user, and
    # an inner fake mode created by Dynamo.  The two phase process is required
    # because the outer fake mode typically won't have a ShapeEnv, even if
    # the user is interested in exporting with dynamic shapes (so the inner
    # fake mode will actually have a ShapeEnv and swap in symbolic sizes.)
    #
    # In this case, it's insufficient to test only one FakeTensor: you need
    # to distinguish between our fake tensor and other fake tensors.  That's
    # what this function does.
    def is_our_fake(self, t):
        return isinstance(t, FakeTensor) and t.fake_mode is self

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert self not in _get_current_dispatch_mode_stack(), func
        try:
            return self.dispatch(func, types, args, kwargs)
        except TypeError:
            log.exception("fake tensor raised TypeError")
            raise

    # No-op if FakeTensorMode is already on the stack
    def __enter__(self):
        if self not in _get_current_dispatch_mode_stack():
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            # no-op
            self.enter_stack.append(False)
            return self

    def __exit__(self, a, b, c):
        live = self.enter_stack.pop()
        if live:
            return super().__exit__(a, b, c)

    def dispatch(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        if func == torch.ops.prim.device.default:
            # NB: Don't use is_our_fake, just serve the fake information
            # as is.  Notice we don't use 'self'; we use args[0].fake_mode
            # because they may not be the same.  It would also be possible
            # to return NotImplemented here, in which case the FakeTensor
            # handler on args[0] would handle it, but we're being nice and
            # short-circuiting quickly.
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "%sFakeTensorMode.__torch_dispatch__: %s", " " * RECURSION_COUNT, func
            )
            incr = IncrementRecursionCount()

        # Some attribute queries that can be serviced directly
        # See Note [is_coalesced is dispatched]
        if func in {
            torch.ops.aten.is_coalesced.default,
            torch.ops.aten.dense_dim.default,
            torch.ops.aten.sparse_dim.default,
        }:
            # NB: no_dispatch is ok here too, this func is very simple
            with in_kernel_invocation_manager(self):
                return func(*args, **kwargs)

        flat_arg_fake_tensors = [
            t
            for t in tree_flatten_only(FakeTensor, (args, kwargs))
            if self.is_our_fake(t)
        ]
        flat_symints = tree_flatten_only(torch.SymInt, (args, kwargs))
        has_symbolic_sizes = (
            any(i._has_symbolic_sizes_strides for i in flat_arg_fake_tensors)
            or len(flat_symints) > 0
        )

        converter = self.fake_tensor_converter

        # To constant propagate through these functions:
        # 1, If this is a lift, the input tensor is guaranteed to be a
        #    constant, so we keep a copy of the original argument along so
        #    we can query it if we're asked to item() it at some later point
        # 2, Some functions that allow Python numbers to bind to Tensors, e.g, torch.div
        if func in self.lift_fns or (
            should_allow_numbers_as_tensors(func)
            and not has_symbolic_sizes
            and not flat_arg_fake_tensors
        ):
            assert all(
                t.constant is not None for t in flat_arg_fake_tensors
            ), "f{func} should not have fake inputs without constants"
            const_args, const_kwargs = pytree.tree_map_only(
                FakeTensor,
                lambda t: t.constant if self.is_our_fake(t) else t,
                (args, kwargs),
            )
            out = func(*const_args, **const_kwargs)
            if type(out) is torch.Tensor and self.may_turn_const(out):
                # NB: not in_kernel_invocation_manager because we're doing real
                # compute here
                # NB: no_dispatch() here is VERY DANGEROUS (like, segfault
                # dangerous) if this is actually a wrapper subclass tensor,
                # therefore the exact type test above
                with no_dispatch():
                    out = out.clone()
                return converter(self, out, make_constant=True)

        # See [subclass inputs] below
        # NB: If you're seeing a mysterious infinite loop involving fake
        # tensor, it might be related to this line.  Though I'm not sure
        # how you'll know to read this comment, as this line won't show up
        # in the stack trace.
        unrecognized_types = self.check_for_subclass(args, kwargs)
        if unrecognized_types:
            not_implemented_log.debug(
                "FakeTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        # if we are in the dispatch mode, we will enter this function even if the inputs
        # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
        # and just support constructors.

        # this is generated from torch.tensor(), which does not use the
        # dispatcher, to allow wrapper subclasses to wrap the new tensor
        if func in self.lift_fns:
            assert (
                len(kwargs) == 0 and len(args) == 1 and type(args[0]) is torch.Tensor
            ), f"{args} {kwargs}"

            return converter(self, args[0])

        # Recompute flat_arg_fake_tensors here again in case some of the inputs
        # were real tensors and fakified in validate_and_convert_non_fake_tensors
        (
            args,
            kwargs,
            flat_arg_fake_tensors,
        ) = self.validate_and_convert_non_fake_tensors(func, converter, args, kwargs)

        # The current constant handling only support tracing systems
        # (aot autograd, torchdynamo) where each operation is run consecutively.
        # Because each operation is run in order, we can trace out and support
        # sequences like: x = torch.tensor(0.); y = x.add_(1)
        # Whenver a constant is written to but with inputs that cannot be evaluated
        # statically, such as random_(), we invalidate all constants that alias the input
        # We will rely on functionalization for use of fake tensors constants as persistent
        # objects on an FX Graph.

        # We dispatch size/stride/numel on the FakeTensor not its constant, so bail on inplace_view
        all_constant = all(e.constant is not None for e in flat_arg_fake_tensors)
        if (
            torch.Tag.nondeterministic_seeded not in func.tags
            and torch.Tag.inplace_view not in func.tags
            and all_constant
            and len(flat_arg_fake_tensors) != 0
            and not has_symbolic_sizes
        ):
            const_args, const_kwargs = pytree.tree_map_only(
                FakeTensor,
                lambda t: t.constant if self.is_our_fake(t) else t,
                (args, kwargs),
            )

            # NB: not in_kernel_invocation_manager(self) as we want to do REAL
            # compute
            with no_dispatch():
                out = func(*const_args, **const_kwargs)

            all_constant = pytree.tree_all_only(
                torch.Tensor, lambda t: self.may_turn_const(t), out
            )

            if all_constant:
                return pytree.tree_map_only(
                    torch.Tensor,
                    lambda t: converter(self, t, make_constant=True),
                    out,
                )

            # we weren't able to turn outputs to constants,
            # so invalidate all constants that might be aliases of the outputs
            for ten in tree_flatten_only(torch.Tensor, out):
                converter.invalidate_constant_aliases(ten)

        # we are falling through to running non constant tensors, any input constant that
        # is written to must be invalidated
        self.invalidate_written_to_constants(func, flat_arg_fake_tensors, args, kwargs)

        # Try for fastpath
        if has_symbolic_sizes:
            fast_impl = get_fast_op_impls().get(func)
            if fast_impl is not None:
                return fast_impl(self, *args, **kwargs)

        # If there's a Python meta, prefer that over the decomposition
        from torch._decomp import meta_table as meta_table

        if func not in meta_table and not self.cpp_meta_supports_symint(func):
            from torch._decomp import decomposition_table

            # Prefer Python decompositions over C++ ones
            if func in decomposition_table and (
                has_symbolic_sizes
                or (
                    # TODO: Remove these exclusions, so that we can remove
                    # this leg entirely
                    torch_decomp_decompositions(func)
                    and all(not e.is_sparse for e in flat_arg_fake_tensors)
                )
            ):
                with self:
                    return decomposition_table[func](*args, **kwargs)

            with self:
                # Decomposes CompositeImplicitAutograd ops
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # prims already wrap FakeTensor inputs to FakeTensor outputs
        # and do device logic, we dont need do anything but run them
        # and ensure that Meta kernels are dispatched to (see)
        # Fake Tensor Dispatch Keys
        # TODO - we should be use the prim aten impl
        # TODO - fix prims complex ops
        if (
            "prims::" in func._schema.name
            and hasattr(func, "prim_meta_impl")
            and not stride_incorrect_op(func)
        ):
            with self:
                return func.prim_meta_impl(*args, **kwargs)

        # Users can register FakeTensor rules for custom operators
        # Call them if they exist.
        if func.name() in torch._custom_op.impl.global_registry:
            custom_op = torch._custom_op.impl.global_registry[func.name()]
            if custom_op is not None and custom_op._has_impl("abstract"):
                ctx = torch._custom_op.impl.AbstractImplCtx(self.shape_env, func)
                with torch._custom_op.impl.set_ctx_getter(lambda: ctx), self:
                    result = custom_op._get_impl("abstract").func(*args, **kwargs)
                    return result

        # special handling for funcs registered through `register_op_impl`,
        # e.g., manipulating args on constructor calls to construct meta tensors
        # and then afterwards wrapping them to a FakeTensor
        for run_impl_check, op_impl in op_implementations:
            if run_impl_check(func):
                op_impl_out = op_impl(self, func, *args, **kwargs)
                if op_impl_out != NotImplemented:
                    return op_impl_out

        def can_run_unsafe_fallback(func: OpOverload):
            if not self.allow_fallback_kernels:
                return False
            # It's OK to try the fallback for built-in ops (e.g. aten, prims)
            # because we control and test these but the fallback leads to unexpected behavior
            # in user-defined custom ops
            #
            # WARNING: DO NOT add any additional namespaces/operators here if they refer to operators
            # outside of the pytorch/pytorch library! Any pre-existing things here
            # are either in the pytorch/pytorch library or have been grandfathered in.
            # The fallback does not always work and MAY CRASH and emit unreadable error messages
            # so it should not be allowed by default.
            allowed_namespaces = {
                "debugprims",
                "prims",
                "aten",
                "xla",
                "vision",
                "torchtext",
                "torchaudio",
            }
            grandfathered_ops_FIXME = {
                "fbgemm::gmm",
            }
            return (
                func.namespace in allowed_namespaces
                or func.name() in grandfathered_ops_FIXME
            )

        def maybe_run_unsafe_fallback(error=None):
            # no meta kernel registered, fallback to kernel for the device
            if has_symbolic_sizes or not can_run_unsafe_fallback(func):
                raise UnsupportedOperatorException(func)
            if error is None:
                error = UnsupportedOperatorException(func)
            return run_fallback_kernel(self, func, args, kwargs, error)

        # Optimization: If there is no Meta kernel, it takes a surprisingly long
        # amount of time to catch the NotImplementedError, so we check it here.
        if not torch._C._dispatch_has_computed_kernel_for_dispatch_key(
            func.name(), "Meta"
        ):
            return maybe_run_unsafe_fallback()

        # run kernel registered to meta for func, which include
        # python meta registrations, prims, decomps, and c++ meta fns (structured kernels)
        # It's possible that the kernel will return NotImplementedError
        try:
            common_device = FakeTensor._find_common_device(func, args, kwargs)
            with in_kernel_invocation_manager(self):
                r = func(*args, **kwargs)
                if (
                    common_device is not None
                    and common_device[0] is not None
                    and is_noncontiguous_supported(common_device[0]) is False
                ):
                    from torch.distributed._tensor.ops.pointwise_ops import(
                        pointwise_ops,
                    )

                    if func in pointwise_ops:
                        r = r.new_empty(r.shape)
        except NotImplementedError as not_implemented_error:
            return maybe_run_unsafe_fallback(not_implemented_error)

        return self.wrap_meta_outputs_with_default_device_logic(r, func, args, kwargs)

    # [subclass inputs]
    # Suppose we enable fake tensor mode.  This means that fake tensor
    # mode will run first.  But what if we do an operation that
    # involves a tensor subclass that will desugar into normal tensor
    # operations?  Without returning NotImplemented, fake tensor mode will run first,
    # decide that a conversion was made (since there was a non fake
    # tensor argument), and report an error that converting non
    # fake tensor is not supported.  What we actually wanted to happen
    # was to give the subclass a chance to figure out what it wants to
    # before erroring out. Returning NotImplemented here allows this.
    def check_for_subclass(self, args, kwargs):
        def check(x):
            return (
                not isinstance(x, FakeTensor)
                and type(x) is not torch.Tensor
                and type(x) is not torch.nn.Parameter
            )

        return [
            type(x) for x in tree_flatten_only(torch.Tensor, (args, kwargs)) if check(x)
        ]

    def validate_and_convert_non_fake_tensors(self, func, converter, args, kwargs):
        """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
        flat_arg_fake_tensors = []

        def validate(x):
            nonlocal flat_arg_fake_tensors
            if not self.is_our_fake(x):
                if torch.Tag.inplace_view in func.tags:
                    raise Exception(
                        f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}"
                    )
                if not self.allow_non_fake_inputs:
                    if isinstance(x, FakeTensor) and x.fake_mode is not self:
                        raise AssertionError("Mixing fake modes NYI")
                    raise Exception(
                        f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode "
                        f"with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}"
                    )

                x = converter(self, x)

            flat_arg_fake_tensors.append(x)
            return x

        args, kwargs = tree_map_only(
            torch.Tensor,
            validate,
            (args, kwargs),
        )
        return args, kwargs, flat_arg_fake_tensors

    def wrap_meta_outputs_with_default_device_logic(self, r, func, args, kwargs):
        wrap = self.gen_wrap_fn(func, args, kwargs)

        # if device is specified, use that
        if kwargs.get("device", None):
            return tree_map(partial(wrap, device=kwargs["device"]), r)

        return tree_map(partial(wrap), r)

    def gen_wrap_fn(self, func, args, kwargs):
        converter = self.fake_tensor_converter

        # Lazily initialized, in case there are no tensor returns
        common_device = None
        has_scalar_only_inputs = False

        def wrap(e, device=None):
            nonlocal common_device
            nonlocal has_scalar_only_inputs

            if isinstance(e, torch.Tensor) and common_device is None:
                (
                    common_device,
                    has_scalar_only_inputs,
                ) = FakeTensor._find_common_device(func, args, kwargs)

            if self.is_our_fake(e):
                torch._check(
                    e.device == common_device,
                    lambda: f"FakeTensor is wrapped to wrong device, found {e.device}, expected {common_device}",
                )

            if (
                isinstance(e, torch.Tensor)
                and not self.is_our_fake(e)
                and converter is not None
            ):
                if has_scalar_only_inputs:
                    # Under FakeTensorMode, op accepts scalar only inputs, such as aten.add/sub/mul/div,
                    # returns a real scalar tensor on CPU. See TensorMeta() in _prims/__init__.py for details.
                    # We thus directly convert real tensor to fake tensor.
                    return converter(self, e)
                else:
                    return converter.from_meta_and_device(
                        self, e, device or common_device
                    )
            else:
                return e

        return wrap

    def cpp_meta_supports_symint(self, func):
        if torch.Tag.view_copy in func.tags:
            return True
        return func in [
            aten.empty.memory_format,
            aten.empty_strided.default,
            aten.as_strided_scatter.default,
            aten.as_strided.default,
            aten.as_strided_.default,
            aten.zeros.default,
            aten.detach.default,
            aten.view_as_real.default,
            aten.view_as_complex.default,
            aten.set_.source_Storage_storage_offset,
            aten._sparse_coo_tensor_with_dims_and_tensors.default,
        ]

    @property
    def lift_fns(self):
        return (aten.lift_fresh.default, aten.lift_fresh_copy.default)

    def may_turn_const(self, t):
        return (
            t.numel() <= CONSTANT_NUMEL_LIMIT
            and not t.is_sparse
            and not self.is_our_fake(t)
            and not t.device.type == "meta"
        )

    def invalidate_written_to_constants(
        self, func, flat_arg_fake_tensors, args, kwargs
    ):
        any_constant = any(e.constant is not None for e in flat_arg_fake_tensors)
        if any_constant and get_schema_info(func).is_mutable():
            schema_info = get_schema_info(func)
            _, new_kwargs = normalize_function(
                func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
            )
            for k, v in new_kwargs.items():
                k = k if (k != "input" or schema_info.has_argument(k)) else "self"
                if (
                    self.is_our_fake(v)
                    and schema_info.is_mutable(k)
                    and v.constant is not None
                ):
                    self.fake_tensor_converter.invalidate_constant_aliases(v.constant)

    def from_tensor(
        self,
        tensor,
        *,
        static_shapes=False,
        ignore_subclass=False,
        source: Optional[Source] = None,
        dynamic_dims: Optional[DimList[DimDynamic]] = None,
        constraint_dims: Optional[DimList[DimConstraint]] = None,
        # Setting this flag will force FakeTensorMode to return `None` if attempting to convert a tensor we have not
        # seen before.
        memoized_only=False,
    ):
        shape_env = self.shape_env
        if static_shapes:
            assert (
                dynamic_dims is None
            ), "cannot set both static_shapes and dynamic_dims"
            shape_env = None
        return self.fake_tensor_converter(
            self,
            tensor,
            shape_env=shape_env,
            ignore_subclass=ignore_subclass,
            source=source,
            dynamic_dims=dynamic_dims,
            constraint_dims=constraint_dims,
            memoized_only=memoized_only,
        )


# NB: returns fake tensors
def run_fallback_kernel(fake_mode, func, args, kwargs, orig_not_implemented_exception):
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception

    inp_impls = {}

    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e):
            if fake_mode.is_our_fake(e):
                out = torch.zeros_like(e, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                inp_impls[id(out)] = e
                return out
            return e

        args = tree_map(to_real_tensor, args)
        kwargs = tree_map(to_real_tensor, kwargs)

        r = func(*args, **kwargs)

    tensor_impls = set()
    storages = set()

    for e in tree_flatten((args, kwargs))[0]:
        if isinstance(e, torch.Tensor):
            if not e.is_sparse:
                storages.add(e._typed_storage()._cdata)

    # TODO: also check metadata change on inputs
    # proper aliasing/metadata relationship between outputs and inputs will
    # not be set up, bc of conversion to device, unless we can reuse an
    # input impl
    for e in tree_flatten(r)[0]:
        if id(e) not in inp_impls and (
            isinstance(e, torch.Tensor)
            and not e.is_sparse
            and e._typed_storage()._cdata in storages
        ):
            raise orig_not_implemented_exception

    def map_out(e):
        if isinstance(e, torch.Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                return fake_mode.fake_tensor_converter(fake_mode, e)
        else:
            return e

    return tree_map(map_out, r)


# Just for use to allow copying a module to fake tensors,
# does not apply elsewhere
class FakeCopyMode(TorchFunctionMode):
    def __init__(self, fake_mode):
        self.fake_mode = fake_mode

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # clone will get called in Parameter deepcopy
        if func == torch._C._TensorBase.clone:
            return func(
                self.fake_mode.from_tensor(args[0], static_shapes=True), **kwargs
            )
        elif func == torch.Tensor.__deepcopy__:
            assert len(args) == 2 and len(kwargs) == 0
            tensor, memo = args

            if id(tensor) in memo:
                return memo[id(tensor)]

            out = self.fake_mode.from_tensor(tensor, static_shapes=True)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

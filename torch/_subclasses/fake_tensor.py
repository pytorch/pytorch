import contextlib
import functools
import itertools
import sys
import weakref
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from torch._ops import OpOverload
from torch._subclasses.meta_utils import MetaConverter, WeakTensorRefKey
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode

from torch.utils._pytree import PyTree, tree_flatten, tree_map

pytree = torch.utils._pytree
T = TypeVar("T")
TensorWeakRef = Any

aten = torch.ops.aten

CONSTANT_NUMEL_LIMIT = 1


@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    reason: str


@dataclass
class DynamicOutputShapeException(RuntimeError):
    func: OpOverload


@dataclass
class DataDependentOutputException(RuntimeError):
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
# structure. Like `MetaConverter`, it uses `WeakTensorRefKey` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverter(object):
    @property
    def tensor_memo(self):
        return self.meta_converter.tensor_memo

    meta_converter: MetaConverter
    constant_storage_mapping: Dict[StorageWeakRef, List[TensorWeakRef]]

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
        if WeakTensorRefKey(t) in self.tensor_memo:
            out = self.tensor_memo[WeakTensorRefKey(t)]
            out._fix_weakref()
            return out
        return None

    def set_tensor_memo(self, t, v):
        th = WeakTensorRefKey(t)

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

    def from_real_tensor(self, fake_mode, t, make_constant=False, shape_env=None):
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
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

        out = self.meta_converter(t, shape_env=shape_env, callback=mk_fake_tensor)
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
    # You must have created the FakeTensorMode with allow_meta == True
    def __call__(self, fake_mode, t, *, make_constant=False, shape_env=None):
        return self.from_real_tensor(fake_mode, t, make_constant, shape_env=shape_env)


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
    # Not in_kernel_invocation_manager as no fake tensor inputs
    with no_dispatch():
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


# _to_copy fails when run with FakeTensors to cuda device
# TODO: debug
@register_op_impl(aten._to_copy.default)
def to_copy(fake_mode, func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    input_device = new_kwargs.pop("device", None)
    out_device = input_device if input_device else new_kwargs["input"].device
    with in_kernel_invocation_manager(fake_mode):
        input = new_kwargs.pop("input").to("meta")
        return FakeTensor(fake_mode, aten._to_copy(input, **new_kwargs), out_device)


# index.Tensor data-dependent in only some conditions
@register_op_impl(
    lambda func: torch.Tag.dynamic_output_shape in func.tags  # type: ignore[attr-defined]
    and func != aten.index.Tensor
)
def dyn_shape(fake_mode, func, *args, **kwargs):
    raise DynamicOutputShapeException(func)


@register_op_impl(
    lambda func: torch.Tag.data_dependent_output in func.tags  # type: ignore[attr-defined]
)
def data_dep(fake_mode, func, *args, **kwargs):
    if fake_mode.throw_on_data_dependent_ops:
        raise DataDependentOutputException(func)
    return NotImplemented


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

    return FakeTensor(fake_mode, out, out_device)


# Dont default to default device handling,
# Since op can take in non-zero sized cpu
# index tensors with cuda self
@register_op_impl(aten.index.Tensor)
def index_tensor(fake_mode, func, *args, **kwargs):
    # dynamic shape op if indices are bool/uint8
    check_no_bool_index_tensors(func, *args, **kwargs)

    return run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)


# takes in multiple-devices, dont default to default device handling
@register_op_impl(aten.index_put.default)
def index_put(fake_mode, func, *args, **kwargs):
    return run_and_return_new_tensor_of_input_device(fake_mode, func, args, kwargs)


# same with index_put, but return the input
@register_op_impl(aten.index_put_.default)
def index_put_(fake_mode, func, *args, **kwargs):
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
        if k == 3 and not kwargs["input"].is_mkldnn and not kwargs["input"].is_xpu:
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
        return torch.Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )

    def __init__(
        self,
        fake_mode,
        elem,
        device: Union[torch.device, str],
        constant: Optional[torch.Tensor] = None,
    ):
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
        # normalize cuda device.
        if device.type == "cuda" and device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.fake_device = device
        self.fake_mode = fake_mode
        self.constant = constant

    @staticmethod
    def from_tensor(t, fake_mode):
        return fake_mode.from_tensor(t)

    # TODO: resolve error in default __repr__
    def __repr__(self):
        with in_kernel_invocation_manager(self.fake_mode):
            self_repr = super().__repr__()
        return f"FakeTensor({self_repr}, {self.fake_device})"

    @classmethod
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
        if any(not issubclass(t, FakeTensor) and t is not torch.Tensor for t in types):
            return NotImplemented

        fake_mode = None
        for arg in itertools.chain(tree_flatten(args)[0], tree_flatten(kwargs)[0]):
            if isinstance(arg, FakeTensor):
                if fake_mode is None:
                    fake_mode = arg.fake_mode
                else:
                    assert fake_mode is arg.fake_mode, "Mixing modes NYI"

        assert fake_mode is not None
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
        if (
            torch._C._should_allow_numbers_as_tensors(
                func.name().split("::")[-1].split(".")[0]
            )
            and common_device is None
        ):
            # ops with scalar only inputs always have result on cpu
            has_scalar_only_inputs = True
            common_device = torch.device("cpu")

        assert common_device is not None, f"Could not find common device for {func}"

        return common_device, has_scalar_only_inputs

    __torch_function__ = torch._C._disabled_torch_function_impl


# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with FakeTensorMode()`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will no be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly incraese.


class FakeTensorMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        allow_fallback_kernels=True,
        allow_meta=False,
        throw_on_data_dependent_ops=True,
        shape_env=None,
    ):
        self.allow_fallback_kernels = allow_fallback_kernels
        self.fake_tensor_converter = FakeTensorConverter()
        self.allow_meta = allow_meta

        # TODO: delete arg and default to true. waiting on dynamo perf regression testing
        self.throw_on_data_dependent_ops = throw_on_data_dependent_ops

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

        self.shape_env = shape_env

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device

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

        flat_arg_fake_tensors = tree_flatten_only(FakeTensor, (args, kwargs))
        flat_symints = tree_flatten_only(torch.SymInt, (args, kwargs))
        has_symbolic_sizes = (
            any([i._has_symbolic_sizes_strides for i in flat_arg_fake_tensors])
            or len(flat_symints) > 0
        )

        converter = self.fake_tensor_converter

        # If this is a lift, the input tensor is guaranteed to be a
        # constant, so we keep a copy of the original argument along so
        # we can query it if we're asked to item() it at some later point
        if func in self.lift_fns:
            out = func(*args, **kwargs)
            if self.may_turn_const(out):
                # NB: not in_kernel_invocation_manager because we're doing real
                # compute here
                with no_dispatch():
                    out = out.clone()
                return converter(self, out, make_constant=True)

        flat_arg_tensors = tree_flatten_only(torch.Tensor, (args, kwargs))
        # See [subclass inputs] below
        # NB: If you're seeing a mysterious infinite loop involving fake
        # tensor, it might be related to this line.  Though I'm not sure
        # how you'll know to read this comment, as this line won't show up
        # in the stack trace.
        if self.check_for_subclass(flat_arg_tensors):
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

        if self.check_for_non_fake(flat_arg_tensors):
            raise Exception(
                "Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. "
                f"Please convert all Tensors to FakeTensors first. Found in {func}(*{args}, **{kwargs})"
            )

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
            torch.Tag.nondeterministic_seeded not in func.tags  # type: ignore[attr-defined]
            and torch.Tag.inplace_view not in func.tags  # type: ignore[attr-defined]
            and all_constant
            and len(flat_arg_fake_tensors) != 0
            and not has_symbolic_sizes
        ):
            const_args, const_kwargs = pytree.tree_map_only(
                FakeTensor, lambda t: t.constant, (args, kwargs)
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

        from torch._decomp import decomposition_table

        with self:
            # Decomposes CompositeImplicitAutograd ops
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

        # IDK: feels bad man, sym_numel on as_strided infinite loops otherwise
        if has_symbolic_sizes and not self.cpp_meta_supports_symint(func):
            from torch._decomp import meta_table as meta_table

            if func == aten.size.default:
                sys.stderr.write(
                    "Trying to call aten.size on a tensor with symbolic shapes. "
                    "It's likely that this is from calling tensor.shape in C++"
                )
                # We do this to allow for better error localization with `TORCH_SHOW_CPP_STACKTRACES=1`
                return None

            with self:
                if func in meta_table:
                    r = meta_table[func](*args, **kwargs)
                    return r
                if func in decomposition_table:
                    return decomposition_table[func](*args, **kwargs)

        if (
            func in decomposition_table
            and torch_decomp_decompositions(func)
            and all(not e.is_sparse for e in flat_arg_fake_tensors)
        ):
            with self:
                return decomposition_table[func](*args, **kwargs)

        if has_symbolic_sizes:
            if not self.cpp_meta_supports_symint(func):
                raise RuntimeError(
                    f"{func} - couldn't find symbolic meta function/decomposition"
                )

        # special handling for funcs registered through `register_op_impl`,
        # e.g., manipulating args on constructor calls to construct meta tensors
        # and then afterwards wrapping them to a FakeTensor
        for run_impl_check, op_impl in op_implementations:
            if run_impl_check(func):
                op_impl_out = op_impl(self, func, *args, **kwargs)
                if op_impl_out != NotImplemented:
                    return op_impl_out

        # run kernel registered to meta for func, which include
        # python meta registrations, prims, decomps, and c++ meta fns (structured kernels)
        try:
            with in_kernel_invocation_manager(self):
                r = func(*args, **kwargs)
        except NotImplementedError as not_implemented_error:
            # no meta kernel registered, fallback to kernel for the device
            if not self.allow_fallback_kernels:
                raise not_implemented_error
            return run_fallback_kernel(self, func, args, kwargs, not_implemented_error)

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
    def check_for_subclass(self, flat_arg_tensors):
        return any(
            not isinstance(x, FakeTensor)
            and type(x) is not torch.Tensor
            and type(x) is not torch.nn.Parameter
            for x in flat_arg_tensors
        )

    def check_for_non_fake(self, flat_arg_tensors):
        return any(
            isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor)
            for x in flat_arg_tensors
        )

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
            if isinstance(e, torch.Tensor) and not isinstance(e, FakeTensor):
                if common_device is None:
                    (
                        common_device,
                        has_scalar_only_inputs,
                    ) = FakeTensor._find_common_device(func, args, kwargs)

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
        if torch.Tag.view_copy in func.tags:  # type: ignore[attr-defined]
            return True
        return func in [
            aten.empty_strided.default,
            aten.as_strided_scatter.default,
            aten.as_strided.default,
            aten.as_strided_.default,
            aten.zeros.default,
            aten.detach.default,
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
            and not isinstance(t, FakeTensor)
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
                    isinstance(v, FakeTensor)
                    and schema_info.is_mutable(k)
                    and v.constant is not None
                ):
                    self.fake_tensor_converter.invalidate_constant_aliases(v.constant)

    def from_tensor(self, tensor, static_shapes=False):
        if static_shapes:
            return self.fake_tensor_converter(self, tensor)
        return self.fake_tensor_converter(self, tensor, shape_env=self.shape_env)


# NB: returns fake tensors
def run_fallback_kernel(fake_mode, func, args, kwargs, orig_not_implemented_exception):
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:  # type: ignore[attr-defined]
        raise orig_not_implemented_exception

    inp_impls = {}

    # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
    # REAL compute (not with meta device)
    with no_dispatch():

        def to_real_tensor(e):
            if isinstance(e, FakeTensor):
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
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)

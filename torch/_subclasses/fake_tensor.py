import contextlib
import functools
import itertools
import weakref
from dataclasses import dataclass
from functools import partial
from typing import Callable, Union

import torch
import torch.fx.experimental.symbolic_shapes as symbolic_shapes
from torch._ops import OpOverload
from torch._subclasses.meta_utils import MetaConverter, WeakTensorRefKey
from torch.fx.operator_schemas import normalize_function
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import enable_torch_dispatch_mode, TorchDispatchMode

from torch.utils._pytree import tree_flatten, tree_map


aten = torch.ops.aten


@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    reason: str


@dataclass
class DynamicOutputShapeException(RuntimeError):
    func: OpOverload


_device_not_kwarg_ops = (
    aten._resize_output_.default,
    aten.nested_tensor.default,
    aten.nested_tensor.out,
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


# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it uses `WeakTensorRefKey` to
# hold a weak reference for all memoized tensors.
class FakeTensorConverter(object):
    tensor_memo: weakref.WeakValueDictionary
    meta_converter: MetaConverter

    def __init__(self):
        # FakeTensors store the FakeTensorMode which in turn stores a
        # FakeTensor, so we need to hold a weak reference to the FakeTensor
        # otherwise we would induce a circular reference
        self.tensor_memo = weakref.WeakValueDictionary()
        self.meta_converter = MetaConverter()

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

    def from_real_tensor(self, fake_mode, t):
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        existing_device = t.device
        # not yet supported in metatensors
        if t.is_quantized:
            raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        with no_dispatch():
            meta_t = self.meta_converter(t)
            if meta_t.device.type != "meta":
                raise UnsupportedFakeTensorException("meta converter nyi")
            out = FakeTensor(fake_mode, meta_t, existing_device)
        if type(t) is torch.nn.Parameter:
            out = torch.nn.Parameter(out, requires_grad=out.requires_grad)  # type: ignore[assignment]
        if t.grad is not None:
            out.grad = self.from_real_tensor(fake_mode, t.grad)
        self.set_tensor_memo(t, out)
        return out

    def from_meta_and_device(self, fake_mode, t, device):
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensor(fake_mode, t, device)
        self.set_tensor_memo(t, out)
        return out

    # There are two ways to call this.  First, you can have manually constructed
    # a meta tensor and you need to turn it into a fake tensor.  In that case,
    # pass a meta tensor and a device argument.  Alternately, you can have a
    # real tensor that you need to convert into a fake tensor; in that case,
    # omit the device.
    #
    # The disallowed case: if you specify the device, it MUST be a meta tensor.
    # However, you're allowed to pass a meta tensor to be turned into a fake
    # tensor; although an odd thing to do, this can occur if you're doing
    # cross ref testing and the inner test is already operating on meta tensors
    def __call__(self, fake_mode, t, device=None):
        if device is None:
            return self.from_real_tensor(fake_mode, t)
        else:
            assert t.device.type == "meta"
            return self.from_meta_and_device(fake_mode, t, device)


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
    r = func(*args, **new_kwargs)
    return fake_mode.fake_tensor_converter(fake_mode, r, out_device)


# Dont default to default device handling,
# since the device of `the_template` is ignored
@register_op_impl(aten.resize_as_.default)
def resize_as_(fake_mode, func, *args, **kwargs):
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
    with no_dispatch():
        input = new_kwargs.pop("input").to("meta")
        return FakeTensor(fake_mode, aten._to_copy(input, **new_kwargs), out_device)


@register_op_impl(aten.clone.default)
def clone(fake_mode, func, input, memory_format=None):
    out_device = input.device
    with no_dispatch():
        out = aten._to_copy(input.to("meta"), memory_format=memory_format)
        return FakeTensor(fake_mode, out, out_device)


# index.Tensor data-dependent in only some conditions
@register_op_impl(
    lambda func: torch.Tag.dynamic_output_shape in func.tags  # type: ignore[attr-defined]
    and func != aten.index.Tensor
)
def data_dep_op(fake_mode, func, *args, **kwargs):
    raise DynamicOutputShapeException(func)


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


# Meta tensors give you the ability to run PyTorch code without having to
# actually do computation through tensors allocated on a `meta` device.
# Because the device is `meta`, meta tensors do not model device propagation.
# FakeTensor extends MetaTensors to also carry an additional `fake_device`
# which tracks devices that would have been used.


@contextlib.contextmanager
def in_kernel_invocation_manager(fake_mode):
    fake_mode.in_kernel_invocation = True
    # See: note [Fake Tensor Dispatch Keys]
    torch._C._add_meta_to_tls_dispatch_include()
    try:
        yield
    finally:
        fake_mode.in_kernel_invocation = False
        torch._C._remove_meta_from_tls_dispatch_include()


class FakeTensor(torch.Tensor):
    fake_device: torch.device
    fake_mode: "FakeTensorMode"
    has_sym_ints: bool

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
    def __new__(cls, fake_mode, elem, device):
        return torch.Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )

    def __init__(self, fake_mode, elem, device: Union[torch.device, str]):
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
        self.has_sym_ints = symbolic_shapes.has_symbolic_sizes_strides(elem)

    @staticmethod
    def from_tensor(t, fake_mode):
        existing_device = t.device
        # TODO: this should use meta converter
        return FakeTensor(fake_mode, t.to(device="meta"), existing_device)

    # TODO: resolve error in default __repr__
    def __repr__(self):
        with in_kernel_invocation_manager(self.fake_mode):
            self_repr = super().__repr__()
        return f"FakeTensor({self.fake_mode}, {self_repr}, {self.fake_device})"

    def stride(self, dim=None):
        if self.has_sym_ints:
            # TODO: As we currently don't support symbolic strides, we'll assume contiguous strides
            # The reason this needs to be here instead of __torch_dispatch__ is that
            # when aten.stride goes into __torch_dispatch__, it expects a list of
            # concrete ints to be returned. So we need to short-circuit that entirely
            strides = symbolic_shapes.create_contiguous(self.shape)
            if dim is None:
                return strides
            else:
                return strides[dim]

        if dim is None:
            return super().stride()
        else:
            return super().stride(dim)

    def new(self, *args, **kwargs):
        # torch.Tensor.new does not go through the normal dispatcher pattern
        # so in order to use the same pattern as normal invocation of
        # returning meta device within the kernel we need to intercept
        # the call here
        # because it doesn't go through the dispatcher, we run into errors
        # when attempting to compute an output in meta, so
        # we compute the real tensor then convert to meta
        out_device = self.fake_device
        with no_dispatch():
            real_out = super().new(*args, **kwargs)

        assert not isinstance(real_out, FakeTensor), real_out
        assert real_out.device.type != "meta", real_out.device

        with no_dispatch():
            meta_out = MetaConverter()(real_out)
            return FakeTensor(self.fake_mode, meta_out, out_device)

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
        # Need this to handle infinite recursion with sparse tensors.
        # Sparse tensors have custom stride policy which means that
        # they will dispatch here on dispatch, and we need to trigger
        # the default behavior.
        # TODO: when we get other tensor types online they will also
        # need to get entries here.
        elif func == torch.ops.aten.stride.default:
            return None

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

        with enable_torch_dispatch_mode(fake_mode):
            return func(*args, **kwargs)

    @staticmethod
    def _find_common_device(func, args, kwargs):
        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite the common_device if it the only existing
        # device comes from a cpu zero-dim tensor
        common_device = None
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

        assert common_device is not None, f"Could not find common device for {func}"

        return common_device

    __torch_function__ = torch._C._disabled_torch_function_impl


# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with torch_enable_mode(FakeTensorMode)`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will no be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly incraese.


class FakeTensorMode(TorchDispatchMode):
    def __init__(self, *, allow_fallback_kernels=True, allow_meta=False):
        self.allow_fallback_kernels = allow_fallback_kernels
        self.fake_tensor_converter = FakeTensorConverter()
        self.allow_meta = allow_meta

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

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            if args[0].fake_mode.in_kernel_invocation:
                return torch.device("meta")
            else:
                return args[0].fake_device
        flat_arg_tensors = [
            i for i in tree_flatten((args, kwargs))[0] if isinstance(i, FakeTensor)
        ]
        flat_symints = [
            i
            for i in tree_flatten((args, kwargs))[0]
            if isinstance(i, torch._C.SymIntNode)
        ]
        has_symbolic_sizes = (
            any([i.has_sym_ints for i in flat_arg_tensors]) or len(flat_symints) > 0
        )
        if has_symbolic_sizes:
            # TODO: Find better approach for this
            # Avoid circular import
            from torch._decomp import decomposition_table
            from torch._meta_registrations import meta_table

            with no_dispatch():
                if symbolic_shapes.is_symbolic_op(func):
                    return symbolic_shapes.handle_symbolic_op(func, args, kwargs)
                if func == aten.size.default:
                    raise RuntimeError(
                        "Trying to call aten.size on a tensor with symbolic shapes. "
                        "It's likely that this is from calling tensor.shape in C++"
                    )

            with self.restore():
                if func in meta_table:
                    r = meta_table[func](*args, **kwargs)
                    return r
                if func in decomposition_table:
                    return decomposition_table[func](*args, **kwargs)

                # Decomposes CompositeImplicitAutograd ops
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # prims already wrap FakeTensor inputs to FakeTensor outputs
        # and do device logic, we dont need do anything but run them
        # and ensure that Meta kernels are dispatched to (see)
        # Fake Tensor Dispatch Keys

        if "prims::" in func._schema.name and len(flat_arg_tensors) != 0:
            try:
                torch._C._add_meta_to_tls_dispatch_include()
                with no_dispatch():
                    return func(*args, **kwargs)
            finally:
                torch._C._remove_meta_from_tls_dispatch_include()

        if has_symbolic_sizes:
            constructors = [aten.empty.memory_format]
            if func not in constructors:
                raise RuntimeError(
                    f"{func} - couldn't find symbolic meta function/decomposition"
                )

        with no_dispatch():
            # TODO: apply as no_dispatch decorator
            converter = self.fake_tensor_converter

            # if we are in the dispatch mode, we will enter this function even if the inputs
            # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
            # and just support constructors. TODO: extend more broadly
            conversion_made = False
            subclass_seen = False

            def check_non_fake_tensor(x):
                nonlocal conversion_made, subclass_seen
                conversion_made = conversion_made or (
                    isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor)
                )
                subclass_seen = subclass_seen or (
                    isinstance(x, torch.Tensor)
                    and not isinstance(x, FakeTensor)
                    and type(x) is not torch.Tensor
                    and type(x) is not torch.nn.Parameter
                )

            tree_map(check_non_fake_tensor, args)
            tree_map(check_non_fake_tensor, kwargs)

            # Suppose we enable fake tensor mode.  This means that fake tensor
            # mode will run first.  But what if we do an operation that
            # involves a tensor subclass that will desugar into normal tensor
            # operations?  Without this line, fake tensor mode will run first,
            # decide that a conversion was made (since there was a non fake
            # tensor argument), and report an error that converting non
            # fake tensor is not supported.  What we actually wanted to happen
            # was to give the subclass a chance to figure out what it wants to
            # before erroring out.  Returning NotImplemented here allows this.
            #
            # NB: If you're seeing a mysterious infinite loop involving fake
            # tensor, it might be related to this line.  Though I'm not sure
            # how you'll know to read this comment, as this line won't show up
            # in the stack trace.
            if subclass_seen:
                return NotImplemented

            # this is generated from torch.tensor(), which does not use the
            # dispatcher, to allow wrapper subclasses to wrap the new tensor
            # we need to handle before error checking
            if func in [
                aten.lift_fresh.default,
                aten.lift_fresh_copy.default,
            ]:
                assert (
                    len(kwargs) == 0
                    and len(args) == 1
                    and type(args[0]) is torch.Tensor
                ), f"{args} {kwargs}"
                with no_dispatch():
                    return converter(self, args[0])

            if conversion_made:
                raise Exception(
                    "Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. "
                    f"Please convert all Tensors to FakeTensors first. Found in {func}(*{args}, **{kwargs})"
                )

            for run_impl_check, op_impl in op_implementations:
                if run_impl_check(func):
                    return op_impl(self, func, *args, **kwargs)

            try:
                with in_kernel_invocation_manager(self):
                    r = func(*args, **kwargs)
            except NotImplementedError as not_implemented_error:
                if not self.allow_fallback_kernels:
                    raise not_implemented_error
                return run_fallback_kernel(
                    self, func, args, kwargs, not_implemented_error
                )

            # TODO: handle non-kwarg devices
            assert func not in _device_not_kwarg_ops, f"NYI: {func}"

            # Lazily initialized, in case there are no tensor returns
            common_device = None

            def wrap(e, device=None):
                nonlocal common_device
                if isinstance(e, torch.Tensor) and not isinstance(e, FakeTensor):
                    if common_device is None:
                        common_device = FakeTensor._find_common_device(
                            func, args, kwargs
                        )
                    return converter(self, e, device or common_device)
                else:
                    return e

            # if device is specified, use that
            if kwargs.get("device", None):
                return tree_map(partial(wrap, device=kwargs["device"]), r)

            return tree_map(partial(wrap), r)

    def from_tensor(self, tensor):
        return self.fake_tensor_converter(self, tensor)


# NB: returns fake tensors
def run_fallback_kernel(fake_mode, func, args, kwargs, orig_not_implemented_exception):
    # these should all be supported, just to be safe
    # avoid fallback for operators which inplace modify metadata
    # because the input fake tensors would be umodified
    if torch.Tag.inplace_view in func.tags:  # type: ignore[attr-defined]
        raise orig_not_implemented_exception

    with no_dispatch():
        inp_impls = {}

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
                    storages.add(e.storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to device, unless we can reuse an
        # input impl
        for e in tree_flatten(r)[0]:
            if id(e) not in inp_impls and (
                isinstance(e, torch.Tensor)
                and not e.is_sparse
                and e.storage()._cdata in storages
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
            return func(self.fake_mode.from_tensor(args[0]), **kwargs)
        elif func == torch.Tensor.__deepcopy__:
            assert len(args) == 2 and len(kwargs) == 0
            tensor, memo = args

            if id(tensor) in memo:
                return memo[id(tensor)]

            out = self.fake_mode.from_tensor(tensor)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)

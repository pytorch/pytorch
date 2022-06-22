import torch

from torch.utils._pytree import tree_map, tree_flatten
from functools import partial
from torch.fx.operator_schemas import normalize_function
from torch.utils._mode_utils import no_dispatch
from torch._subclasses.meta_utils import MetaConverter
from typing import Union, Callable
from torch._ops import OpOverload
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode, enable_torch_dispatch_mode
import weakref
import functools
import itertools
from dataclasses import dataclass


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
    aten.pin_memory.default,
    aten.is_pinned.default,
    aten.to.device,
    aten.to.prim_Device,
    aten._pin_memory.default,
    aten._resize_output.functional,
    aten._resize_output.out,
)

# this op is never actually used
_non_kwarg_device_constructors = (torch.ops.aten._list_to_tensor,)


def contains_tensor_types(type):
    tensor_type = torch._C.TensorType.get()
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )


_like_tensor_constructors = (
    aten.empty_like.default,
    aten.full_like.default,
    aten.ones_like.default,
    aten.rand_like.default,
    aten.randn_like.default,
    aten.randint_like.default,
    aten.randint_like.low_dtype,
    aten.randn_like.default,
    aten.zeros_like.default,
    aten.new_empty.default,
    aten.new_empty_strided.default,
    aten.new_full.default,
    aten.new_zeros.default,
    aten.new_ones.default,
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
# structure. Like `MetaConverter`, it will keep alive all
# tensors that are converted to FakeTensors.
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
        if t in self.tensor_memo:
            out = self.tensor_memo[t]
            out._fix_weakref()
            return out
        return None

    def from_real_tensor(self, fake_mode, t):
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        existing_device = t.device
        # not yet supported in metatensors
        if t.is_complex():
            raise UnsupportedFakeTensorException("complex nyi in meta tensors")
        if t.is_sparse:
            raise UnsupportedFakeTensorException("sparse nyi in meta tensors")
        if t.is_quantized:
            raise UnsupportedFakeTensorException("quantized nyi in meta tensors")
        with no_dispatch():
            out = FakeTensor(fake_mode, self.meta_converter(t), existing_device)
        if type(t) is torch.nn.Parameter:
            out = torch.nn.Parameter(out, requires_grad=out.requires_grad)  # type: ignore[assignment]
        self.tensor_memo[t] = out
        return out

    def from_meta_and_device(self, fake_mode, t, device):
        maybe_memo = self._get_memo(t)
        if maybe_memo is not None:
            return maybe_memo
        out = FakeTensor(fake_mode, t, device)
        self.tensor_memo[t] = out
        return out

    def __call__(self, fake_mode, t, device=None):
        assert t.device.type != "meta" or device is not None
        if t.device.type != "meta":
            return self.from_real_tensor(fake_mode, t)
        else:
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

@register_op_impl(lambda func: (_is_tensor_constructor(func) or func in _like_tensor_constructors))
def contructors(fake_mode, func, *args, **kwargs):
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


# _to_copy fails when run with FakeTensors to cuda device
# TODO: debug
@register_op_impl(torch.ops.aten._to_copy.default)
def to_copy(fake_mode, func, *args, **kwargs):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    input_device = new_kwargs.pop("device", None)
    out_device = input_device if input_device else new_kwargs["input"].device
    with no_dispatch():
        input = new_kwargs.pop("input").to("meta")
        return FakeTensor(
            fake_mode, torch.ops.aten._to_copy(input, **new_kwargs), out_device
        )

@register_op_impl(torch.ops.aten.clone.default)
def clone(fake_mode, func, input, memory_format=None):
    out_device = input.device
    with no_dispatch():
        out = torch.ops.aten._to_copy(input.to("meta"), memory_format=memory_format)
        return FakeTensor(fake_mode, out, out_device)

# index.Tensor data-dependent in only some conditions
@register_op_impl(lambda func: torch.Tag.dynamic_output_shape in func.tags  # type: ignore[attr-defined]
                  and func != aten.index.Tensor)
def data_dep_op(fake_mode, func, *args, **kwargs):
    raise DynamicOutputShapeException(func)

# Bool Indices get Expanded as Masks
# See: IndexingUtils.h:expandTensors
def check_no_bool_index_tensors(func, self, indices):
    for index in indices:
        if index is not None and index.dtype in (torch.bool, torch.uint8):
            raise DynamicOutputShapeException(func)

# Meta tensors give you the ability to run PyTorch code without having to
# actually do computation through tensors allocated on a `meta` device.
# Because the device is `meta`, meta tensors do not model device propagation.
# FakeTensor extends MetaTensors to also carry an additional `fake_device`
# which tracks devices that would have been used.


class FakeTensor(torch.Tensor):
    fake_device: torch.device
    fake_mode: "FakeTensorMode"

    @staticmethod
    def __new__(cls, fake_mode, elem, device):
        return torch.Tensor._make_subclass(
            cls, elem, elem.requires_grad, dispatch_device=True
        )

    def __init__(self, fake_mode, elem, device: Union[torch.device, str]):
        # elem does not need to be recorded, because FakeTensor *is a* elem
        assert elem.device.type == "meta"
        device = device if isinstance(device, torch.device) else torch.device(device)
        assert device.type != "meta"
        self.fake_device = device
        self.fake_mode = fake_mode

    @staticmethod
    def from_tensor(t, fake_mode):
        existing_device = t.device
        return FakeTensor(fake_mode, t.to(device="meta"), existing_device)

    # TODO: resolve error in default __repr__
    def __repr__(self):
        return f"FakeTensor({self.fake_device}, {self.size()}, {self.dtype})"

    def new(self, *args, **kwargs):
        # torch.Tensor.new does not go through the normal dispatcher pattern
        # so in order to use the same pattern as normal invocation of
        # returning meta device within the kernel we need to intercept
        # the call here
        out_device = self.fake_device
        if "device" in kwargs:
            kwarg_device = kwargs.pop("device")
            out_device = kwarg_device if kwarg_device else out_device
            kwargs["device"] = "meta"
        self.in_kernel_invocation = True
        try:
            with no_dispatch():
                meta_out = super().new(*args, **kwargs)
        finally:
            self.in_kernel_invocation = False

        with no_dispatch():
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
            raise Exception(
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
    def __init__(self, allow_cpu_fallback=True):
        self.allow_cpu_fallback = allow_cpu_fallback
        self.fake_tensor_converter = FakeTensorConverter()

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

        # prims already wrap FakeTensor inputs to FakeTensor outputs
        # and do device logic, we dont need do anything but run them
        if "prims::" in func._schema.name:
            with no_dispatch():
                return func(*args, **kwargs)

        with no_dispatch():
            # TODO: apply as no_dispatch decorator
            converter = self.fake_tensor_converter

            # this is generated from torch.tensor(), which does not use the
            # dispatcher, to allow wrapper subclasses to wrap the new tensor
            # we need to handle before error checking
            if func == torch.ops.aten.lift.default:
                assert (
                    len(kwargs) == 0
                    and len(args) == 1
                    and type(args[0]) is torch.Tensor
                )
                with no_dispatch():
                    return converter(self, args[0])

            def wrap(e, device=None):
                if isinstance(e, torch.Tensor) and not isinstance(e, FakeTensor):
                    return converter(self, e, device)
                else:
                    return e

            # if we are in the dispatch mode, we will enter this function even if the inputs
            # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
            # and just support constructors. TODO: extend more broadly
            conversion_made = False

            def check_non_fake_tensor(x):
                nonlocal conversion_made
                conversion_made = conversion_made or (
                    isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor)
                )

            tree_map(check_non_fake_tensor, args)
            tree_map(check_non_fake_tensor, kwargs)

            if conversion_made:
                raise Exception(
                    "Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. "
                    f"Please convert all Tensors to FakeTensors first. Found in {func}"
                )

            for run_impl_check, op_impl in op_implementations:
                if run_impl_check(func):
                    return op_impl(self, func, *args, **kwargs)

            if func == aten.index.Tensor:
                check_no_bool_index_tensors(func, *args, **kwargs)

            self.in_kernel_invocation = True
            try:
                r = func(*args, **kwargs)
            except NotImplementedError as not_implemented_error:
                if not self.allow_cpu_fallback:
                    raise not_implemented_error
                r = run_cpu_fallback(func, args, kwargs, not_implemented_error)
            finally:
                self.in_kernel_invocation = False

            # TODO: handle non-kwarg devices
            assert func not in _device_not_kwarg_ops, f"NYI: {func}"

            # if device is specified, use that
            if kwargs.get("device", None):
                return tree_map(partial(wrap, device=kwargs["device"]), r)

            common_device = FakeTensor._find_common_device(func, args, kwargs)

            return tree_map(partial(wrap, device=common_device), r)

    def from_tensor(self, tensor):
        return self.fake_tensor_converter(self, tensor)

def run_cpu_fallback(func, args, kwargs, orig_not_implemented_exception):
    with no_dispatch():
        def to_cpu(e):
            if isinstance(e, FakeTensor):
                return torch.zeros_like(e, device="cpu")
            return e

        try:
            args = tree_map(to_cpu, args)
            kwargs = tree_map(to_cpu, kwargs)

            r = func(*args, **kwargs)
        except Exception as new_exception:
            raise orig_not_implemented_exception from new_exception

        tensor_impls = set()
        storages = set()

        for e in tree_flatten((args, kwargs))[0]:
            if isinstance(e, torch.Tensor):
                tensor_impls.add(e)
                storages.add(e.storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to cpu, error on reused impls
        for e in tree_flatten(r)[0]:
            if e in tensor_impls or (
                isinstance(e, torch.Tensor) and e.storage()._cdata in storages
            ):
                raise orig_not_implemented_exception

    # we're only converting these to MetaTensors now, not Fake Tensors,
    # and the cpu inputs should be temporary. just convert outputs to meta
    # and continue
    return tree_map(MetaConverter(), r)


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

import torch

from torch.utils._pytree import tree_map
from functools import partial
from torch.fx.operator_schemas import normalize_function
from torch.utils._mode_utils import no_dispatch
from torch._subclasses.meta_utils import MetaConverter
from typing import Union, Callable
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
import functools
import contextlib

aten = torch.ops.aten

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
)

# TODO: use tags
_inplace_view_ops = (
    aten.rename_.default,
    aten.as_strided_.default,
    aten.detach_.default,
    aten.squeeze_.default,
    aten.squeeze_.dim,
    aten.squeeze_.dimname,
    aten.t_.default,
    aten.transpose_.default,
    aten.unsqueeze_.default,
    aten.swapaxes_.default,
    aten.swapdims_.default,
    aten.resize_.default,  # needs inplace_view tag
    aten._resize_output_.default,  # needs inplace_view tag
    aten.set_.default,  # needs inplace-metadata tag
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

cpu_fallback_enabled = False

@contextlib.contextmanager
def enable_cpu_fallback(enable_cpu_fallback: bool):
    global cpu_fallback_enabled
    orig = cpu_fallback_enabled
    cpu_fallback_enabled = enable_cpu_fallback
    try:
        yield
    finally:
        cpu_fallback_enabled = orig

# Similar to `MetaConverter`, this is a class for converting
# multiple tensors into fake tensors which share the same view/storage
# structure. Like `MetaConverter`, it will keep alive all
# tensors that are converted to FakeTensors.
class FakeTensorConverter(MetaConverter):
    def __init__(self):
        self.tensor_memo = {}
        self.meta_converter = MetaConverter()
        # we need to throw on inplace-view operations on FakeTensors
        # that are created from real tensors, because we have no way
        # affecting the original tensor without ruining the "simulated"
        # property of FakeTensorMode
        self.fake_from_real_set = set()

    def from_real_tensor(self, t):
        existing_device = t.device
        self.tensor_memo[t] = FakeTensor(self.meta_converter(t), existing_device)
        self.fake_from_real_set.add(self.tensor_memo[t])
        return self.tensor_memo[t]

    def from_meta_and_device(self, t, device):
        if t in self.tensor_memo:
            return self.tensor_memo[t]
        self.tensor_memo[t] = FakeTensor(t, device)
        return self.tensor_memo[t]

    def __call__(self, t, device=None):
        assert t.device.type != 'meta' or device is not None
        if t in self.tensor_memo:
            return self.tensor_memo[t]
        elif t.device.type != 'meta':
            return self.from_real_tensor(t)
        else:
            return self.from_meta_and_device(t, device)

def run_cpu_fallback(func, args, kwargs, orig_not_implemented_exception):
    with no_dispatch():
        def to_cpu(e):
            if isinstance(e, FakeTensor):
                return torch.empty_like(e, device="cpu")
            return e
        try:
            args = tree_map(to_cpu, args)
            kwargs = tree_map(to_cpu, kwargs)
            r = func(*args , **kwargs)
        except Exception:
            # original error more orinformative
            raise orig_not_implemented_exception
        tensor_impls = set()

        def collect_impls(e):
            if isinstance(e, torch.Tensor):
                tensor_impls.add(e)

        tree_map(collect_impls, (args, kwargs))
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to cpu, error on reused impls

        def throw_on_reused_impls(e):
            if e in tensor_impls:
                raise orig_not_implemented_exception

        tree_map(throw_on_reused_impls, r)

    # we're only converting these to MetaTensors now, not Fake Tensors,
    # and the cpu inputs should be temporary. just convert outputs to meta
    # and continue
    return tree_map(MetaConverter(), r)

op_handlers = []

def register_op_handler(run_handler_check: Union[Callable[[OpOverload], bool], OpOverload]):
    def handler_decorater(op_handler):
        global op_handlers
        if isinstance(run_handler_check, OpOverload):
            op_handlers.append((lambda func: func == run_handler_check, op_handler))
        else:
            op_handlers.append((run_handler_check, op_handler))

        return op_handler

    return handler_decorater

# This classes virtualizes .device() calls, need to short-circuit
# it insteead of calling device again or we would keep on recurring
# NB: register this first, since device will be called frequently
@register_op_handler(torch.ops.prim.device.default)
def device_op(cls_or_mode_instance, func, types, args, kwargs, run_function, converter):
    assert len(args) == 1 and isinstance(args[0], FakeTensor)
    return args[0].fake_device


# _to_copy fails when run with FakeTensors to cuda device
# TODO: debug
@register_op_handler(torch.ops.aten._to_copy.default)
def to_copy(cls_or_mode_instance, func, types, args, kwargs, run_function, converter):
    _, new_kwargs = normalize_function(
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )
    out_device = new_kwargs.pop("device", new_kwargs["input"].device)
    with no_dispatch():
        input = new_kwargs.pop("input").to("meta")
        return FakeTensor(
            torch.ops.aten._to_copy(input, **new_kwargs), out_device
        )

@register_op_handler(lambda func: _is_tensor_constructor(func) or func in _like_tensor_constructors)
def contructors(cls_or_mode_instance, func, types, args, kwargs, run_function, converter):
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

    out_device = new_kwargs.pop("device", default_device)
    new_kwargs["device"] = torch.device("meta")
    r = run_function(func, types, args, new_kwargs)
    return FakeTensor(r, out_device)

@register_op_handler(lambda func: func in (aten.to.prim_Device, aten.to.device))
def non_kwarg_to(cls_or_mode_instance, func, types, args, kwargs, run_function, converter):
    _, new_kwargs = normalize_function(func, args, kwargs, normalize_to_only_use_kwargs=True)
    input_device = new_kwargs["device"]
    out_device = input_device if input_device else new_kwargs["input"].device
    new_kwargs["device"] = torch.device("meta")
    r = run_function(func, types, (), new_kwargs)
    return converter(r, out_device)

# Dont default to default device handling,
# since the device of `the_template` is ignored
@register_op_handler(aten.resize_as_.default)
def resize_as_(cls_or_mode_instance, func, types, args, kwargs, run_function, converter):
    return run_function(func, types, args, kwargs)

# Meta tensors give you the ability to run PyTorch code without having to
# actually do computation through tensors allocated on a `meta` device.
# Because the device is `meta`, meta tensors do not model device propagation.
# FakeTensor extends MetaTensors to also carry an additional `fake_device`
# which tracks devices that would have been used.

def torch_dispatch_impl(cls_or_mode_instance, func, types, args, kwargs, run_function):
    kwargs = kwargs if kwargs else {}
    in_fake_mode = isinstance(cls_or_mode_instance, FakeTensorMode)
    converter = cls_or_mode_instance.fake_tensor_converter if in_fake_mode else FakeTensorConverter()


    def wrap_to_fake(e, device=None):
        if isinstance(e, torch.Tensor) and not isinstance(e, FakeTensor):
            return converter.from_real_tensor(e)
        else:
            return e

    # if we are in the dispatch mode, we will enter this function even if the inputs
    # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
    # and just support constructors. TODO: extend more broadly
    if in_fake_mode:
        args, kwargs = tree_map(wrap_to_fake, (args, kwargs))

    def check_for_input_tensor_inplace_view(e):
        if e in converter.fake_from_real_set:
            raise Exception(f"Inplace metadata change on a fake tensor that was created from a real tensor input."
                            f"Convert all tensor inputs to fake tensors and re-run.{func}")

    if func in _inplace_view_ops:
        tree_map(check_for_input_tensor_inplace_view, (args, kwargs))
    if kwargs.get("out", None) and func.overload_name == "out":
        check_for_input_tensor_inplace_view(kwargs["out"])


    for run_op_handler_check, op_handler in op_handlers:
        if run_op_handler_check(func):
            return op_handler(cls_or_mode_instance, func, types, args, kwargs, run_function, converter)

    try:
        r = run_function(func, types, args, kwargs)
    except NotImplementedError as not_implemented_error:
        if not cpu_fallback_enabled:
            raise not_implemented_error
        r = run_cpu_fallback(func, args, kwargs, not_implemented_error)

    # TODO: handle non-kwarg devices
    assert func not in _device_not_kwarg_ops, f"NYI: {func}"

    def wrap(e, device=None):
        if isinstance(e, torch.Tensor) and not isinstance(e, FakeTensor):
            return converter(e, device)
        else:
            return e


    # if device is specified, use that
    if kwargs.get("device", None):
        return tree_map(partial(wrap, device=kwargs["device"]), r)

    common_device = FakeTensor._find_common_device(func, args, kwargs)

    return tree_map(partial(wrap, device=common_device), r)


class FakeTensor(torch.Tensor):
    fake_device: torch.device

    @staticmethod
    def __new__(cls, elem, device):
        return torch.Tensor._make_subclass(
            cls, elem, elem.requires_grad, dispatch_device=True
        )

    def __init__(self, elem, device: Union[torch.device, str]):
        # elem does not need to be recorded, because FakeTensor *is a* elem
        assert elem.device.type == "meta"
        device = device if isinstance(device, torch.device) else torch.device(device)
        assert device.type != "meta"
        self.fake_device = device

    @staticmethod
    def from_tensor(t):
        existing_device = t.device
        return FakeTensor(t.to(device="meta"), existing_device)

    # TODO: resolve error in default __repr__
    def __repr__(self):
        return f"FakeTensor({self.fake_device})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def run_fn(func, types, args, kwargs):
            return torch.Tensor.__torch_dispatch__(func, types, args, kwargs)
        return torch_dispatch_impl(cls, func, types, args, kwargs, run_fn)

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
    def __init__(self):
        self.fake_tensor_converter = FakeTensorConverter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def run_fn(func, types, args, kwargs):
            return func(*args, **kwargs)
        return torch_dispatch_impl(self, func, types, args, kwargs, run_fn)

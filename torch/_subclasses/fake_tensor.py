import torch

from torch.utils._pytree import tree_map
from functools import partial
from torch.fx.operator_schemas import normalize_function
from torch.utils._mode_utils import no_dispatch
from typing import Union
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
import functools

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


# Meta tensors give you the ability to run PyTorch code without having to
# actually do computation through tensors allocated on a `meta` device.
# Because the device is `meta`, meta tensors do not model device propagation.
# FakeTensor extends MetaTensors to also carry an additional `fake_device`
# which tracks devices that would have been used.

def torch_dispatch_impl(cls_or_mode_instance, func, types, args, kwargs, run_function):
    kwargs = kwargs if kwargs else {}

    # This classes virtualizes .device() calls, need to short-circuit
    # it instead of calling device again or we would keep on recurring
    if func == torch.ops.prim.device.default:
        assert len(args) == 1 and isinstance(args[0], FakeTensor)
        return args[0].fake_device

    def wrap(e, device=None):
        if isinstance(e, torch.Tensor) and not isinstance(e, FakeTensor):
            if device:
                return FakeTensor(e, device)
            else:
                return FakeTensor.from_tensor(e)
        else:
            return e

    # if we are in the dispatch mode, we will enter this function even if the inputs
    # are not FakeTensors. For now, throw if any non-Fake Tensor inputs
    # and just support constructors. TODO: extend more broadly
    if isinstance(cls_or_mode_instance, FakeTensorMode):
        conversion_made = False

        def check_non_fake_tensor(x):
            nonlocal conversion_made
            conversion_made = conversion_made or (isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor))

        tree_map(check_non_fake_tensor, args)
        tree_map(check_non_fake_tensor, kwargs)

        if conversion_made:
            raise Exception(
                "Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. "
                f"Please convert all Tensors to FakeTensors first. Found in {func}"
            )

    # _to_copy fails when run with FakeTensors to cuda device
    # TODO: debug
    if func == torch.ops.aten._to_copy.default:
        _, new_kwargs = normalize_function(
            func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )
        out_device = new_kwargs.pop("device", new_kwargs["input"].device)
        with no_dispatch():
            input = new_kwargs.pop("input").to("meta")
            return FakeTensor(
                torch.ops.aten._to_copy(input, **new_kwargs), out_device
            )

    if _is_tensor_constructor(func):
        assert func not in _non_kwarg_device_constructors
        _, new_kwargs = normalize_function(
            func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )
        # cpu is default device if none is specified
        out_device = new_kwargs.pop("device", torch.device("cpu"))
        new_kwargs["device"] = torch.device("meta")
        r = run_function(func, types, (), new_kwargs)
        return FakeTensor(r, out_device)

    r = run_function(func, types, args, kwargs)

    # TODO: handle non-kwarg devices
    assert func not in _device_not_kwarg_ops, f"NYI: {func}"

    # if device is specified, use that
    if kwargs.get("device", None):
        return tree_map(partial(wrap, device=kwargs["device"]), r)

    # operators which copy size from another tensor do not
    # also take device from the size tensor
    # other size_as operators are not builtin operators
    if func == aten.resize_as_.default:
        _, new_kwargs = normalize_function(
            func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
        )
        # device of the input is returned
        return tree_map(partial(wrap, device=new_kwargs["input"].device), r)

    common_device = FakeTensor._find_common_device(func, args, kwargs)

    return tree_map(partial(wrap, device=common_device), r)


class FakeTensor(torch.Tensor):
    fake_device: torch.device

    @staticmethod
    def __new__(cls, elem, device):
        return torch.Tensor._make_subclass(cls, elem, elem.requires_grad, dispatch_device=True)

    def __init__(self, elem, device: Union[torch.device, str]):
        # elem does not need to be recorded, because FakeTensor *is a* elem
        assert elem.device.type == "meta"
        device = device if isinstance(device, torch.device) else torch.device(device)
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

class FakeTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def run_fn(func, types, args, kwargs):
            return func(*args, **kwargs)
        return torch_dispatch_impl(self, func, types, args, kwargs, run_fn)

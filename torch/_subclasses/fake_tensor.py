import torch

from torch._subclasses import BaseTensor
from torch.utils._pytree import tree_map
from functools import partial
from torch.fx.operator_schemas import normalize_function
from torch.utils._mode_utils import no_dispatch
from typing import Union

aten = torch.ops.aten

_device_not_kwarg_ops = (
    aten._resize_output_.default,
    aten.nested_tensor.default,
    aten.pin_memory.default,
    aten.is_pinned.default,
    aten.to.device,
    aten.to.prim_Device,
    aten._pin_memory.default,
)

# Meta tensors give you the ability to run PyTorch code without having to
# actually do computation through tensors allocated on a `meta` device.
# Because the device is `meta`, meta tensors do not model device propagation.
# FakeTensor extends MetaTensors to also carry an additional `fake_device`
# which tracks devices that would have been used.


class FakeTensor(BaseTensor):
    fake_device: torch.device

    @staticmethod
    def __new__(cls, elem, device):
        return super().__new__(cls, elem, dispatch_device=True)

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
        kwargs = kwargs if kwargs else {}

        # This classes virtualizes .device() calls, need to short-circuit
        # it insteead of calling device again or we would keep on recurring
        if func == torch.ops.prim.device.default:
            assert len(args) == 1 and isinstance(args[0], FakeTensor)
            return args[0].fake_device

        # Run the original computation

        # _to_copy fails when run with FakeTensors to cuda device
        # TODO: debug
        if func == torch.ops.aten._to_copy.default:
            _, new_kwargs = normalize_function(
                func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
            )
            out_device = new_kwargs.pop("device", new_kwargs["input"].device)
            with no_dispatch():
                input = new_kwargs.pop("input").to("meta")
                return FakeTensor(torch.ops.aten._to_copy(input, **new_kwargs), out_device)

        r = super().__torch_dispatch__(func, types, args, kwargs)

        def wrap(e, device):
            # inplace ops can return fake tensors
            if isinstance(e, torch.Tensor) and not isinstance(e, cls):
                return FakeTensor(e, device)
            else:
                return e

        # TODO: handle non-kwarg devices
        assert func not in _device_not_kwarg_ops, f"NYI: {func}"
        assert (
            func != aten._pin_memory.default and func != aten.pin_memory.default
        ), f"NYI: {func}"

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

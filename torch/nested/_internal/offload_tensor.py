from typing import *  # noqa: F403
import weakref

import torch
from torch.nested._internal.utils import _try_get_fake_mode
from torch.utils import _pytree as pytree


# Don't create these directly, use make_offload_tensor so that we can track them.
class OffloadTensor(torch.Tensor):
    device_tensor: Optional[torch.Tensor]
    host_tensor: Optional[torch.Tensor]

    @staticmethod
    def __new__(
        cls,
        device: Optional[torch.device] = None,
        device_tensor: Optional[torch.Tensor] = None,
        host_tensor: Optional[torch.Tensor] = None,
        offload_hook: Optional[Callable] = None,
    ):
        source = device_tensor if device_tensor is not None else host_tensor
        assert source is not None
        device = device_tensor.device if device_tensor is not None else device
        assert device is not None
        shape = source.shape
        kwargs = {}
        kwargs["strides"] = source.stride()
        kwargs["storage_offset"] = source.storage_offset()
        kwargs["device"] = device
        kwargs["layout"] = source.layout
        kwargs["requires_grad"] = source.requires_grad
        kwargs["dtype"] = source.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(
        self,
        device: Optional[torch.device] = None,
        device_tensor: Optional[torch.Tensor] = None,
        host_tensor: Optional[torch.Tensor] = None,
        offload_hook: Optional[Callable] = None,
    ):
        self.device_tensor = device_tensor
        self.host_tensor = host_tensor
        self.offload_hook = offload_hook

    def __repr__(self):
        source_tensor = (
            self.device_tensor if self.device_tensor is not None else self.host_tensor
        )
        host_is_cached = self.host_tensor is not None
        # TODO(soulitzer): improve this
        return f"OffloadTensor({repr(source_tensor)}, host_is_cached={host_is_cached})"

    def offload(self):
        # Mutating the subclass field is fine because we are not tracing
        if self.host_tensor is None:
            assert self.device_tensor is not None
            self.host_tensor = self.device_tensor.to("cpu", non_blocking=True)
            if self.offload_hook is not None:
                ret = self.offload_hook(self.host_tensor, self.device_tensor)
                assert ret is None
        self.device_tensor = None
        return self.host_tensor

    def restore(self):
        if self.device_tensor is None:
            assert self.host_tensor is not None
            # Do not cache the device_tensor to avoid mutating subclass fields
            # during compile.
            ret = self.host_tensor.to(self.device, non_blocking=True)
            return ret
        else:
            return self.device_tensor

    def is_offloaded(self):
        return self.device_tensor is None

    def __tensor_flatten__(self):
        ctx = {
            "device": self.device,
        }
        # We specialize on the offloadedness of input tensors to the compiled
        # region and trace the restore logic into the graph.
        inner_tensors = []
        if self.device_tensor is not None:
            inner_tensors.append("device_tensor")
        if self.host_tensor is not None:
            inner_tensors.append("host_tensor")
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        device_tensor = inner_tensors.get("device_tensor", None)
        host_tensor = inner_tensors.get("host_tensor", None)
        device = meta["device"]
        return create_offload_tensor(
            device=device, device_tensor=device_tensor, host_tensor=host_tensor
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # Doing any operation on a CachedTensor automatically calls restore and then unwraps
        kwargs = {} if kwargs is None else kwargs
        unwrapped_args = pytree.tree_map_only(
            OffloadTensor, lambda x: x.restore(), args
        )
        unwrapped_kwargs = pytree.tree_map_only(
            OffloadTensor, lambda x: x.restore(), kwargs
        )
        out = func(*unwrapped_args, **unwrapped_kwargs)

        return out

torch.serialization.add_safe_globals([OffloadTensor])


_global_is_pending_offload = False


def request_offload_all():
    # Before the next creation of a new offload tensor, we want to offload all
    # existing offload tensors.
    # If you request offload all in the compiled region, the offloading will
    # happen after compiled region is done.
    global _global_is_pending_offload
    _global_is_pending_offload = True


class OffloadTensorRegistry:
    def __init__(self, offload_hook: Optional[Callable] = None):
        self.wrapper_refs = []
        # Upon execution, the hook
        # hook(host_tensor, device_tensor) -> None
        self.offload_hook = None

    def create(self, device, device_tensor, host_tensor):
        ret = OffloadTensor(
            device=device,
            device_tensor=device_tensor,
            host_tensor=host_tensor,
            offload_hook=self.offload_hook,
        )
        self.wrapper_refs.append(weakref.ref(ret))
        return ret

    def maybe_offload_all(self):
        global _global_is_pending_offload

        if _global_is_pending_offload:
            still_alive = []
            for ref in self.wrapper_refs:
                if (t := ref()) is not None:
                    t.offload()
                    still_alive.append(weakref.ref(t))
            self.wrapper_refs = still_alive
            _global_is_pending_offload = False
        return

    def copy(self):
        # Is this what we want to do?
        ret = OffloadTensorRegistry()
        ret.wrapper_refs = self.wrapper_refs.copy()


_global_offload_tensor_registry = None


def init_offload_tensor_registry(registry):
    global _global_offload_tensor_registry
    _global_offload_tensor_registry = registry


def create_offload_tensor(device, device_tensor, host_tensor):
    assert _global_offload_tensor_registry is not None
    if _try_get_fake_mode(device_tensor if device_tensor is not None else host_tensor):
        return OffloadTensor(
            device=device,
            device_tensor=device_tensor,
            host_tensor=host_tensor,
        )
    else:
        return _global_offload_tensor_registry.create(
            device, device_tensor, host_tensor
        )


def maybe_offload_all(device_tensor):
    assert _global_offload_tensor_registry is not None
    if _try_get_fake_mode(device_tensor):
        # Do nothing here. Offloading is an eager-only optimization.
        return
    else:
        _global_offload_tensor_registry.maybe_offload_all()


# Offload tensor wrapper is disposable
def _make_offload_tensor(
    device_tensor: Optional[torch.Tensor] = None,
    host_tensor: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> "OffloadTensor":
    assert device_tensor is not None or host_tensor is not None
    assert device is not None or device_tensor is not None
    maybe_offload_all(device_tensor if device_tensor is not None else host_tensor)
    ret = create_offload_tensor(
        device=device, device_tensor=device_tensor, host_tensor=host_tensor
    )
    return ret

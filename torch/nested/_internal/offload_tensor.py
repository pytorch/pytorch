from typing import *  # noqa: F403
import weakref

import torch
from torch.nested._internal.utils import _try_get_fake_mode
from torch.utils import _pytree as pytree
from torch.utils.weak import WeakTensorKeyDictionary


__all__ = ["make_offload_tensor", "request_offload_all"]


class TensorRegistry:
    def __init__(self):
        self._tensor_to_id = WeakTensorKeyDictionary()
        self._id_to_tensor = dict()
        self._next_id = 0

    def register(self, t, t_id=None):
        if not t_id:
            t_id = self._next_id
            self._next_id += 1
        self._tensor_to_id[t] = t_id
        self._id_to_tensor[t_id] = weakref.ref(t)
        return t_id

    def try_get_tensor(self, id: int):
        ref = self._id_to_tensor.get(id)
        if ref is None:
            return None
        if (t := ref()) is None:
            del self._id_to_tensor[id]
            return None
        return t

    def try_get_int(self, tensor: torch.Tensor):
        return self._tensor_to_id.get(tensor)

    def get_int(self, tensor: torch.Tensor):
        if (ret := self.try_get_int(tensor)) is None:
            return self.register(tensor)
        return ret


# Make sure dynamo doesn't try to trace through this
def register_tensor(t, t_id=None):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    if isinstance((t := mb_unwrap_functional_tensor(t)), FakeTensor):
        return t.register_nested_int_id(t_id)
    else:
        return _global_tensor_registry.register(t, t_id=t_id)


def try_get_int(t):
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor

    if isinstance((t := mb_unwrap_functional_tensor(t)), FakeTensor):
        return t.try_get_nested_int_id()
    else:
        return _global_tensor_registry.try_get_int(t)


# short id
def sid(t):
    if t is None:
        return "None"
    return id(t) % 10000


_global_tensor_registry = TensorRegistry()


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
    ):
        self.device_tensor = device_tensor
        self.host_tensor = host_tensor

    def __repr__(self):
        source_tensor = (
            self.device_tensor if self.device_tensor is not None else self.host_tensor
        )
        host_is_cached = self.host_tensor is not None
        # TODO(soulitzer): improve this
        return f"OffloadTensor({repr(source_tensor)}, host_is_cached={host_is_cached})"

    # Once a tensor has been offloaded it can never be restored
    def offload(self):
        # users should not call into this
        # Mutating the tensor's fields is fine because we are not in tracing.
        if self.host_tensor is None:
            assert self.device_tensor is not None
            self.host_tensor = self.device_tensor.to("cpu", non_blocking=True)
            register_tensor(self.host_tensor, try_get_int(self.device_tensor))
            print("offload", sid(self.device_tensor), " -> ", sid(self.host_tensor))
        else:
            print(
                "offload (cached)",
                sid(self.device_tensor),
                " -> ",
                sid(self.host_tensor),
            )
        self.device_tensor = None
        return self.host_tensor

    def restore(self):
        if self.device_tensor is None:
            assert self.host_tensor is not None
            # It would be nice to cache the device tensor upon restore, but not sure
            # we can mutate fields of a subclass during compile.
            # in this case we
            ret = self.host_tensor.to(self.device, non_blocking=True)
            print("restore", sid(self.host_tensor), " -> ", sid(ret))
            return ret

        else:
            print(
                "restore (no-op)",
                sid(self.host_tensor),
                " -> ",
                sid(self.device_tensor),
            )
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


_global_is_pending_offload = False


def request_offload_all():
    # Before the next creation of a new offload tensor, we want to offload all
    # existing offload tensors.
    # If you request offload all in the compiled region, the offloading will
    # happen after compiled region is done.
    global _global_is_pending_offload
    _global_is_pending_offload = True


class OffloadTensorRegistry:
    def __init__(self):
        self.wrapper_refs = []

    def create(self, device, device_tensor, host_tensor):
        ret = OffloadTensor(
            device=device,
            device_tensor=device_tensor,
            host_tensor=host_tensor,
        )
        # TODO(soulitzer): Who is responsible for registering the tensors for the first time?
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


_global_offload_tensor_registry = OffloadTensorRegistry()


def create_offload_tensor(device, device_tensor, host_tensor):
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
    if _try_get_fake_mode(device_tensor):
        # Do nothing here. Offloading is an eager-only optimization.
        return
    else:
        _global_offload_tensor_registry.maybe_offload_all()


# Offload tensor wrapper is disposable
@torch._dynamo.allow_in_graph
def make_offload_tensor(
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

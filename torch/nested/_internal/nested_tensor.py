import torch
from typing import Tuple
from torch._C import DispatchKey, DispatchKeySet
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403

_tensor_id_counter = 0
_tensor_id_registry = WeakTensorKeyDictionary()

def get_tensor_id(tensor):
    global _tensor_id_counter
    if tensor not in _tensor_id_registry:
        _tensor_id_registry[tensor] = _tensor_id_counter
        _tensor_id_counter += 1
    return torch._C._get_singleton_int(_tensor_id_registry[tensor])

class NestedTensor(torch.Tensor):
    buffer: torch.Tensor
    offsets: torch.Tensor
    is_jagged: bool
    _size: Tuple[int, int, int]
    nb_tensors: int

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        buffer,
        *,
        nested_sizes=None,
        offsets=None,
        nb_tensors=None,
        **kwargs
    ):
        _kwargs = {}
        _kwargs["dtype"] = buffer.dtype
        _kwargs["device"] = buffer.device
        _kwargs["requires_grad"] = kwargs.get("requires_grad", False)
        _kwargs["dispatch_sizes_strides_policy"] = "sizes"
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)
        _kwargs["_extra_dispatch_keys"] = ks
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, (0,), **_kwargs)
        if r.requires_grad:
            raise ValueError(
                "buffer should not require grad when constructing NestedTensor")
        r.buffer = buffer
        return r

    def __init__(
        self,
        buffer,
        *,
        nested_sizes=None,
        offsets=None,
        nb_tensors=None,
        **kwargs
    ):
        super().__init__()

        # Only support jagged for now.
        self.is_jagged = True

        assert nested_sizes is None
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(buffer, NestedTensor)
        assert buffer.ndim == 2
        assert nb_tensors is not None
        assert nb_tensors == offsets.shape[0] - 1

        # In a later PR, we'll need to accept an additional size argument
        # to handle dynamic shapes.
        ragged_dim = get_tensor_id(offsets)
        D = buffer.shape[1]
        B = offsets.shape[0] - 1
        self._size = (B, ragged_dim, D)
        self.offsets = offsets
        self._nested_sizes = None
        self.nb_tensors = nb_tensors
        return

    def __repr__(self):
        # We should implement this in torch/_tensor_str.py instead
        grad_fn_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        if self.grad_fn :
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self.offsets}{grad_fn_str})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        any_jagged = any([is_jagged(x) for x in args])
        any_strided = any([is_strided(x) for x in args])

        if any_jagged and any_strided:
            raise NotImplementedError

        if any_jagged:
            # Lazy import to avoid circular dependency
            from .ops import lookup_jagged
            fn = lookup_jagged(func, *args, **kwargs)
            if fn is not None:
                return fn(*args, **kwargs)

        raise NotImplementedError

def is_jagged(x: torch.Tensor) -> bool:
    return isinstance(x, NestedTensor) and x.is_jagged

def is_strided(x: torch.Tensor) -> bool:
    return isinstance(x, NestedTensor) and not x.is_jagged

# Not actually a view!
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):
        ctx.save_for_backward(x.offsets)
        ctx.nb_tensors = x.nb_tensors
        return x.buffer

    @staticmethod
    def backward(ctx, gO: torch.Tensor):
        offsets, = ctx.saved_tensors
        nb_tensors = ctx.nb_tensors
        return NestedTensor(gO, offsets=offsets, nb_tensors=nb_tensors)

# Not actually a view!
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer: torch.Tensor, offsets: torch.Tensor, nb_tensors: int):
        return NestedTensor(buffer, offsets=offsets, nb_tensors=nb_tensors)

    @staticmethod
    def backward(ctx, gO: torch.Tensor):
        return gO.buffer, None, None

# Need to make it obvious that users should be passing in offsets
def jagged_from_list(
        tensors: Sequence[torch.Tensor],
        offsets: Optional[torch.Tensor]) -> Tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors
    """
    assert len(set(t.dtype for t in tensors)) == 1
    assert len(set(t.device for t in tensors)) == 1
    assert all(t.ndim == 2 for t in tensors)
    assert len(set(t.shape[1] for t in tensors)) == 1

    lengths = torch.tensor([t.shape[0] for t in tensors])
    _offsets = torch.cat([torch.tensor([0]), lengths.cumsum(0)])
    if offsets is not None:
        assert torch.all(offsets == _offsets).item()
    else:
        offsets = _offsets

    return ViewNestedFromBuffer.apply(torch.cat(tensors, dim=0), offsets, len(tensors)), offsets

def buffer_from_jagged(jagged):
   return ViewBufferFromNested.apply(jagged)

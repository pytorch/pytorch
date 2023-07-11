import contextlib
from typing import Optional

import warnings
import torch
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
    _pop_torch_dispatch_stack, _push_on_torch_dispatch_stack, DispatchKey


# TODO: Limitations and things about enable_torch_dispatch_mode we should fix before exposing it:
# - We need a better user-facing api for _DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)

class TorchDispatchMode:
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """
    def __init__(self, _dispatch_key=None):
        if _dispatch_key is not None:
            assert isinstance(_dispatch_key, torch._C.DispatchKey)
            self.__dict__['_dispatch_key'] = _dispatch_key

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()

    def __enter__(self):
        _push_mode(self, self.__dict__.get("_dispatch_key", None))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode(self.__dict__.get("_dispatch_key", None))

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn("`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`")
        instance = cls(*args, **kwargs)
        return instance

def _get_current_dispatch_mode():
    stack_len = _len_torch_dispatch_stack()
    # Return a user mode on the stack if there are any
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    # Check our proxy mode slot
    mb_proxy = torch._C._get_proxy_tensor_mode()
    if mb_proxy is not None:
        return mb_proxy
    # Check our fake mode slot
    return torch._C._get_fake_tensor_mode()


def _get_current_dispatch_mode_stack():
    stack_len = _len_torch_dispatch_stack()
    user_modes = [_get_dispatch_stack_at(i) for i in range(stack_len)]
    mb_proxy = [] if torch._C._get_proxy_tensor_mode() is None else [torch._C._get_proxy_tensor_mode()]
    mb_fake = [] if torch._C._get_fake_tensor_mode() is None else [torch._C._get_fake_tensor_mode()]
    return user_modes + mb_proxy + mb_fake

def _push_mode(mode, k: Optional[DispatchKey] = None):
    if k is not None:
        from torch._ops import push_mode_for_key, get_cached_ops
        # See Note [Not Caching Per-Dispatch-Key Mode Handlers]
        # Clear the cache of every op that has been used so far, for this particular key.
        ks = torch._C._functionality_to_backend_keys(k)
        for op in get_cached_ops():
            for key in ks:
                op._uncache_dispatch(key)
        push_mode_for_key(k, mode)
    else:
        _push_on_torch_dispatch_stack(mode)


def _pop_mode(k: Optional[DispatchKey] = None):
    if k is not None:
        from torch._ops import pop_mode_for_key
        return pop_mode_for_key(k)
    else:
        return _pop_torch_dispatch_stack()


@contextlib.contextmanager
def _pop_mode_temporarily(k: Optional[DispatchKey] = None):
    old = _pop_mode(k)
    try:
        yield old
    finally:
        _push_mode(old, k)


@contextlib.contextmanager
def _disable_current_modes():
    mode_len = _len_torch_dispatch_stack()
    old_modes = [_pop_mode() for _ in range(mode_len)]

    # Manually disable proxy and fake modes, if any are active
    mb_proxy = [] if torch._C._get_proxy_tensor_mode() is None else [torch._C._unset_proxy_tensor_mode()]
    mb_fake = [] if torch._C._get_fake_tensor_mode() is None else [torch._C._unset_fake_tensor_mode()]
    try:
        yield old_modes + mb_proxy + mb_fake
    finally:
        for mode in reversed(old_modes):
            _push_mode(mode)
        if mb_proxy:
            torch._C._set_proxy_tensor_mode(mb_proxy[0])
        if mb_fake:
            torch._C._set_fake_tensor_mode(mb_fake[0])


class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

def supports_mode_tracing(t):
    # In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,
    # It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.
    tensor_like = isinstance(t, torch.Tensor) and type(t) != torch.Tensor
    return tensor_like and hasattr(t, "__tensor_flatten__") and hasattr(t, "__tensor_unflatten__")

def transform_subclass(t, callback):
    assert supports_mode_tracing(t)
    # convert the tensor subclass into its constituent dense tensors,
    # and apply a transformation to each dense tensor.
    from torch.utils._pytree import tree_map_only
    flattened_tensors, ctx = type(t).__tensor_flatten__(t)
    transformed_tensors = tree_map_only(torch.Tensor, callback, flattened_tensors)
    return type(t).__tensor_unflatten__(transformed_tensors, ctx)

# TODO: this.
#def set_subclass_output_aliasing(func, outputs

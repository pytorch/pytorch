import contextlib
from typing import Iterator, Set
import functools

import warnings
from torch.utils._mode_utils import _enable_mode_checks, _ModeInfo
from torch._C import _get_torch_dispatch_mode, _set_torch_dispatch_mode
from dataclasses import dataclass


_cur_torch_dispatch_mode = []

@dataclass
class TorchDispatchModeInfo(_ModeInfo):
    def __init__(self):
        super().__init__(mode_name="torch_dispatch", mode_class=TorchDispatchMode)

    def get_mode(self):
        return _get_torch_dispatch_mode()

    def set_mode(self, mode):
        return _set_torch_dispatch_mode(mode)


# TODO: Limitations and things about enable_torch_dispatch_mode we should fix before exposing it:
# - We need a better user-facing api for torch._C._DisableTorchDispatch that
#   is able to selectively disable __torch_dispatch__ of a particular class.
# - It doesn't work with the tensor constructors (torch.tensor, torch.Tensor)
# - Better name (see https://github.com/pytorch/pytorch/pull/63496#discussion_r694091694)
@contextlib.contextmanager
def enable_torch_dispatch_mode(mode, *, replace=None, ignore_preexisting=False) -> Iterator[None]:
    """
    Context manager that causes all pytorch operators to dispatch to the passed-in
    type's __torch_dispatch__ function, including operations that accept no tensors
    but return a tensor.

    This function is non-compositional; if there is already an existing mode,
    it will raise an error

    This function is safe to use inside a ``__torch_dispatch__`` mode handler,
    as the mode is guaranteed to be disabled in this context.  You can use
    this context manager to reinstate the mode so that calls to overridable
    APIs recursively call back into your mode handler (this can easily cause
    infinite loops, so use with care!)

    enable_torch_dispatch_mode is affected by _DisableTorchDispatch.

    Args:
        mode (:class:`TorchDispatchMode`, Tensor-like class, or None): the
            mode to set as current mode.  If you pass a Tensor-like class,
            it will be treated as a non-compositional mode with no state,
            which is convenient if you have an existing tensor subclass
            that you'd like to apply globally in a quick and dirty way.
            Passing None will disable the current mode.
        replace (:class:`TorchDispatchMode` or Tensor-like class): the
            mode to replace.  You can use this argument to change the mode in
            a situation where you know what the current mode is (and you are
            intentionally overwriting it.)  If you don't know what the current
            mode is, use ``ignore_preexisting`` instead.
        ignore_preexisting (bool): if True, ignore any preexisting mode
            and overwrite it with the passed mode.
    """
    old = _get_torch_dispatch_mode()
    if old is mode:
        yield mode
        return

    _enable_mode_checks(mode, mode_info=TorchDispatchModeInfo(), replace=replace, ignore_preexisting=ignore_preexisting)

    _set_torch_dispatch_mode(mode)
    try:
        yield mode  # type: ignore[misc]
    finally:
        # for enable mode, the current mode _must_ have been None before
        _set_torch_dispatch_mode(old)


def _wrap_torch_dispatch(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if isinstance(f, classmethod):
            raise RuntimeError("TorchDispatchMode's torch_dispatch function " +
                               "should be a normal method not a class method")
        inner = _cur_torch_dispatch_mode.pop() if len(_cur_torch_dispatch_mode) > 0 else None

        try:
            with enable_torch_dispatch_mode(inner):
                return f(self, *args, **kwargs)
        finally:
            assert _get_torch_dispatch_mode() is None
            _cur_torch_dispatch_mode.append(inner)

    return wrapped


# Implementation note, since this is based on TorchFunctionMode, this had the
# same dilemma: I had a choice about how much of mode stacks
# to implement in Python versus in C++.  At time of writing, I did not care
# too much about implementation efficiency; however, I do care about making it
# hard for users to implement modes in the wrong way.  In the end, it turned
# out to be possible to implement mode stacks entirely from userland, with the
# C++ API providing only _get_torch_dispatch_mode() and
# _set_torch_dispatch_mode(), so I opted to provide some unsafe C++ bindings and
# have the bulk of the logic for managing the stack in Python, which helped
# simplify the C++ API surface.  It would also have been valid to build in the
# notion of mode stack directly into C++ but in this design it's substantially
# more difficult to interact with TorchDispatchModeMeta.

class TorchDispatchModeMeta(type):
    """
    Metaclass for :class:`TorchDispatchMode`; it:
        * Reenables the inner mode, so that by default PyTorch API calls
          will compositionally proceed to the next mode on the stack.

    The default behavior for this is important, as it is easy to
    accidentally write ``_wrap_torch_dispatch`` implementations that are not
    compositional, and the wrapping here makes the obvious code do the
    right thing (aka, this is why there is a metaclass).
    """
    def __new__(metacls, name, bases, dct):
        if '__torch_dispatch__' in dct:
            dct['__torch_dispatch__'] = _wrap_torch_dispatch(dct['__torch_dispatch__'])
        return super().__new__(metacls, name, bases, dct)


class TorchDispatchMode(metaclass=TorchDispatchModeMeta):
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
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()

    def __enter__(self):
        old = _get_torch_dispatch_mode()
        if self in _cur_torch_dispatch_mode or self is old:
            raise RuntimeError(f"{self} is already active in the mode stack")
        _cur_torch_dispatch_mode.append(old)
        _set_torch_dispatch_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_torch_dispatch_mode(_cur_torch_dispatch_mode.pop() if len(_cur_torch_dispatch_mode) > 0 else None)

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn("`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`")
        instance = cls(*args, **kwargs)
        return instance


class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

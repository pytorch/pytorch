import contextlib

import warnings
from torch.utils._mode_utils import _ModeInfo
from torch._C import _get_torch_dispatch_mode, _set_torch_dispatch_mode
from dataclasses import dataclass
from typing import List


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
# more difficult to interact with it.
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
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError()

    def __enter__(self):
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode()

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn("`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`")
        instance = cls(*args, **kwargs)
        return instance


# WARNING: If you access this list to see all the active modes in order, do not update the list in place
# since it alters the actual mode stack
_cur_torch_dispatch_mode: List[TorchDispatchMode] = []


def get_current_dispatch_mode():
    return _cur_torch_dispatch_mode[-1] if len(_cur_torch_dispatch_mode) > 0 else None


def _push_mode(mode):
    if len(_cur_torch_dispatch_mode) == 0:
        _set_torch_dispatch_mode(_TorchDispatchStackMode())
    _cur_torch_dispatch_mode.append(mode)


def _pop_mode():
    assert len(_cur_torch_dispatch_mode) > 0
    old = _cur_torch_dispatch_mode.pop()
    if len(_cur_torch_dispatch_mode) == 0:
        _set_torch_dispatch_mode(None)
    return old


@contextlib.contextmanager
def _pop_mode_temporarily():
    old = _pop_mode()
    try:
        yield old
    finally:
        _push_mode(old)

# a helper "mode" used by the torch dispatch push helper method. This is the only mode that will ever
# be active at the C++ level and it will run the current mode
class _TorchDispatchStackMode:
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        with _pop_mode_temporarily() as old:
            if len(_cur_torch_dispatch_mode) > 0:
                _set_torch_dispatch_mode(self)
            # we can't check the type of __torch_function__ here but this is sufficient for checking it's a classmethod
            if old.__torch_dispatch__.__self__ is type(old):
                raise RuntimeError("TorchFunctionMode's torch_function function " +
                                   "should be a normal method not a class method")
            return old.__torch_dispatch__(func, types, args, kwargs)

class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)

import functools
import torch
from typing import Iterator, TypeVar
from dataclasses import dataclass
from contextlib import contextmanager

T = TypeVar('T')

# This file has all the logic to dedupe logic between TorchDispatchMode and
# TorchFunctionMode
#
# Specifically, it has the helper functions for enable_ and context manager and the
# ModeInfo class, which is extended by each where they are different


# in order to dedupe the logic between TorchDispatchMode and TorchFunctionMode, this
# is a container to hold all the differences between the modes. Then functions like
# _enable_mode are able to use this container to call functions or get correctly
# formatted names
@dataclass
class _ModeInfo:
    mode_name: str
    mode_class: type  # the class related to the mode that's allowed to be passed in

    def mode_class_name(self):
        return self.mode_class.__name__

    def get_mode(self):
        """gets the current mode for this type of mode"""
        raise NotImplementedError()

    def set_mode(self, mode):
        """
        set mode to for this type of mode. Note that no checks are done on this, it's the unsafe
        version where checks are assumed to have been already done by the helper function
        """
        raise NotImplementedError()


# shared version of enable_torch_function/enable_torch_dispatch_mode in order to deduplicate the code.
# The differences between the modes are captured by `mode_info` and then queried when they're
# needed during the function's invocation
def _enable_mode_checks(mode: T, mode_info: _ModeInfo, *, replace=None, ignore_preexisting=False) -> Iterator[T]:
    if not (
        mode is None or
        isinstance(mode, mode_info.mode_class) or
        (isinstance(mode, type) and not issubclass(mode, mode_info.mode_class))
    ):
        raise ValueError(f'expected to get {mode_info.mode_class_name()}, Tensor-like class, '
                         f'or None as an argument got {type(mode)} instead')
    old = mode_info.get_mode()
    if replace is not None and old is not replace:
        raise ValueError(f"Wanted to replace {replace} but current mode is {old}")
    elif old is not None and not ignore_preexisting and old is not replace :
        if isinstance(mode, mode_info.mode_class):
            help_text = f'Use `with Mode():` instead.'
        else:
            help_text = (
                'If you intended to completely override the preexisting mode, '
                'pass ignore_preexisting=True.  This can result in unexpected '
                'behavior; please consider rewriting your mode to be a subclass '
                f'of {mode_info.mode_class_name()} to make it compositional!'
            )
        raise ValueError(
            f'Attempted to enable_{mode_info.mode_name}_mode, but there is already an '
            f'active mode {old}.  {help_text}'
        )
    # NB: we don't require TorchFunctionMode/TorchDispatchMode since this is intended to also
    # let you directly pass a Tensor subclass type to "mode-ify" it.
    if mode is not None:
        required_fn = "__" + mode_info.mode_name + "__"
        if not hasattr(mode, required_fn):
            raise ValueError(
                f'The argument passed to enable_{mode_info.mode_name}_mode must implement {required_fn}'
            )


# returns if all are the same mode
def all_same_mode(modes):
    return all(tuple(mode == modes[0] for mode in modes))

@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard

import torch
from typing import TypeVar
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

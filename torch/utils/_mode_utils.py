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

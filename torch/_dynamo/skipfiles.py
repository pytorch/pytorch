import _collections_abc
import _weakrefset
import abc
import collections
import contextlib
import copy
import copyreg
import dataclasses
import enum
import functools
import importlib
import inspect
import linecache
import logging
import multiprocessing
import operator
import os
import posixpath
import random
import re
import selectors
import signal
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import weakref

import torch

try:
    import torch._prims

    # isort: split
    # TODO: Hack to unblock simultaneous landing changes. Fix after https://github.com/pytorch/pytorch/pull/81088 lands
    import torch._prims.utils
    import torch._prims.wrappers
    import torch._refs
    import torch._refs.nn
    import torch._refs.nn.functional
    import torch._refs.special

    HAS_PRIMS_REFS = True
except ImportError:
    HAS_PRIMS_REFS = False

from . import config

"""
A note on skipfiles:

Dynamo consults this file to determine whether code should be compiled or skipped.

A skip applies at the frame boundary, meaning dynamo either triggers a graph break
at the beginning of the frame or attempts to trace the whole frame.  When skipping
a frame, recursively called frames are still traced by dynamo unless also skipped.

Skipfiles (skipped at the file level instead of function level) still apply on a
frame-by-frame boundary as dynamo traces, but apply to all functions in that file.

@skip is a helper decorator that can be applied to your function to cause it to be
included here.
"""


def _strip_init_py(s):
    return re.sub(r"__init__.py$", "", s)


def _module_dir(m: types.ModuleType):
    return _strip_init_py(m.__file__)


SKIP_DIRS = [
    # torch.*
    _module_dir(torch),
    # torchdynamo.*
    os.path.dirname(__file__) + "/",
    "<frozen importlib",
    "<__array_function__ internals>",
] + [
    # skip some standard libs
    _module_dir(m)
    for m in (
        abc,
        collections,
        contextlib,
        copy,
        copyreg,
        dataclasses,
        enum,
        functools,
        importlib,
        inspect,
        linecache,
        logging,
        multiprocessing,
        operator,
        os,
        posixpath,
        random,
        re,
        selectors,
        signal,
        tempfile,
        threading,
        tokenize,
        traceback,
        types,
        typing,
        unittest,
        weakref,
        _collections_abc,
        _weakrefset,
    )
]
FILENAME_ALLOWLIST = {
    torch.nn.Sequential.__init__.__code__.co_filename,
    torch.set_rng_state.__code__.co_filename,
}

# Include optimizer code for tracing
FILENAME_ALLOWLIST |= set(
    [
        inspect.getfile(obj)
        for obj in torch.optim.__dict__.values()
        if inspect.isclass(obj)
    ]
)

FILENAME_ALLOWLIST |= {torch.optim._functional.__file__}

if HAS_PRIMS_REFS:
    FILENAME_ALLOWLIST |= {
        torch._prims.__file__,
        torch._prims.utils.__file__,
        torch._prims.wrappers.__file__,
        torch._refs.__file__,
        torch._refs.special.__file__,
        torch._refs.nn.functional.__file__,
    }

FILENAME_ALLOWLIST |= {torch.optim._functional.__file__}

SKIP_DIRS_RE = None


def _recompile_re():
    global SKIP_DIRS_RE
    SKIP_DIRS_RE = re.compile(f"^({'|'.join(map(re.escape, SKIP_DIRS))})")


def add(import_name: str):
    if isinstance(import_name, types.ModuleType):
        return add(import_name.__name__)
    assert isinstance(import_name, str)
    module_spec = importlib.util.find_spec(import_name)
    if not module_spec:
        return
    origin = module_spec.origin
    if origin is None:
        return
    global SKIP_DIRS_RE
    SKIP_DIRS.append(_strip_init_py(origin))
    _recompile_re()


def check(filename, allow_torch=False):
    """Should skip this file?"""
    if filename is None:
        return True
    if filename in FILENAME_ALLOWLIST:
        return False
    if allow_torch and is_torch(filename):
        return False
    return bool(SKIP_DIRS_RE.match(filename))


# skip common third party libs
for _name in (
    "functorch",
    "intel_extension_for_pytorch",
    "networkx",
    "numpy",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "onnx_tf",
    "pandas",
    "sklearn",
    "tabulate",
    "tensorflow",
    "tensorrt",
    "torch2trt",
    "tqdm",
    "tree",
    "tvm",
    "fx2trt_oss",
    "xarray",
):
    add(_name)

_recompile_re()


def is_torch_inline_allowed(filename):
    return any(
        filename.startswith(_module_dir(mod))
        for mod in config.skipfiles_inline_module_allowlist
    )


@functools.lru_cache(None)
def dynamo_dir():
    return _module_dir(importlib.import_module(config.dynamo_import))


def is_torch(filename):
    if filename.startswith(dynamo_dir()):
        return False
    return filename.startswith(_module_dir(torch))

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
from typing import Optional, TYPE_CHECKING

import torch
import torch._inductor.test_operators
import torch.distributed
import torch.utils._content_store
from .utils import getfile

from .variables.functions import (
    NestedUserFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)


"""
A note on skipfiles:

Dynamo consults this file to determine whether function should be inlined or skipped.

A skip applies at the frame boundary, meaning dynamo either triggers a graph break
at the beginning of the frame or attempts to trace/inline the whole frame. When skipping
a frame, recursively called frames are still traced by dynamo unless also skipped.

Skipfiles (skipped at the file level instead of function level) still apply on a
frame-by-frame boundary as dynamo traces, but apply to all functions in that file.

@skip is a helper decorator that can be applied to your function to cause it to be
included here.

Dynamo skip/inline rules & priorities are defined as follows:
* Inline is the default behavior and will be used unless explicitly skipped.
* Dynamo has two SKIPLIST: BUILTIN_SKIPLIST and THIRDPARTY_SKIPLIST.
    * BUILTIN_SKIPLIST contains builtin python modules, such as abc, collections, etc.
    * THIRDPARTY_SKIPLIST contains common third party libraries, such as numpy, pandas, etc.
* Functions in these two SKIPLISTs are always skipped, except when they are explicitly
    put into the two INLINELIST: FILENAME_INLINELIST and SUBMODULE_INLINELIST.
* PyTorch(torch) is in the BUILTIN_SKIPLIST by default, but there are many cases
    where we want inline the functions under torch namespace. We should add them
    into FILENAME_INLINELIST or SUBMODULE_INLINELIST to make dynamo inline those functions.
* If you call functions under skipped modules/files, Dynamo will wrap these functions
    as SkipFilesVariable. There are a few functions(e.g, collections.OrderedDict) that
    we have special handling at SkipFilesVariable.call_function.

Overall: *_INLINELIST has precedence over *_SKIPLIST has precedence over DEFAULT (inline)

To figure out what the behavior is, check the following list in order:
* FILENAME_INLINELIST (Inline if YES)
* SUBMODULE_INLINELIST (Inline if YES)
* BUILTIN_SKIPLIST & THIRDPARTY_SKIPLIST (Skip if YES)
* Inline by default

"""


BUILTIN_SKIPLIST = (
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
    torch,  # torch/* is skipped by default unless specified in FILENAME_INLINELIST or SUBMODULE_INLINELIST
    traceback,
    types,
    typing,
    unittest,
    weakref,
    _collections_abc,
    _weakrefset,
)

# third party libraries skiplist is defined by str, because users may not use these libraries.
# we should use lazy import & skip in the future.
THIRDPARTY_SKIPLIST = (
    "functorch",
    "fx2trt_oss",
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
    "xarray",
)


def _strip_init_py(s):
    return re.sub(r"__init__.py$", "", s)


def _module_dir(m: types.ModuleType):
    return _strip_init_py(m.__file__)


FUNC_INLINELIST = {
    "torch.autograd.function.once_differentiable",
}


# Force inline functions in these files, even the files is in *_SKIPLIST.
FILENAME_INLINELIST = {
    "torch.nn.modules.container",
    "torch.random",
    "torch._inductor.test_operators",
    "torch.utils._content_store",
    "torch._dynamo.external_utils",
    "torch._dynamo.comptime",
    "torch._dynamo.polyfill",
    "torch.optim._functional",
    "torch.utils._foreach_utils",
    "torch.ao.quantization.pt2e.qat_utils",
    "torch.ao.quantization.quantizer.xnnpack_quantizer",
    "torch.ao.quantization.pt2e.representation.rewrite",
    "torch.ao.quantization.pt2e.utils",
    "torch.ao.quantization.pt2e.eval_utils",
    "torch._dynamo._trace_wrapped_higher_order_op",
    "torch._export.constraints",
    "torch._export.db.examples",
    "torch._export.wrappers",
    "torch._higher_order_ops.cond",
    "torch._functorch.apis",
    "torch._functorch.deprecated",
    "torch.distributed.tensor.parallel._utils",
    "torch.distributed.tensor.parallel.style",
    "torch.distributed.tensor.parallel._data_parallel_utils",
    "torch.distributed._tensor.api",
    "torch.distributed._tensor.device_mesh",
}


if torch.distributed.is_available():
    # Inline the checkpoint code from distributed
    FILENAME_INLINELIST |= {
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper"
    }

# Include optimizer code for tracing
FILENAME_INLINELIST |= {
    str(obj.__module__) for obj in torch.optim.__dict__.values() if inspect.isclass(obj)
}


# Force inline functions under these modules, even the modules is in *_SKIPLIST.
SUBMODULE_INLINELIST = {
    "torch.nn",
    "torch.distributions",
    "torch.testing",
    "torch.ao.nn",
    "torch._refs",
    "torch._prims",
    "torch._decomp",
    "torch.utils._contextlib",
    "torch.utils._pytree",
    "torch.fx._pytree",
    "torch.sparse",
}


if torch.distributed.is_available():
    SUBMODULE_INLINELIST.add("torch.distributed._functional_collectives")


@functools.lru_cache(None)
def get_func_inlinelist():
    inlinelist = set()
    for f in FUNC_INLINELIST:
        inlinelist.add(eval(f))
    return inlinelist


@functools.lru_cache(None)
def get_filename_inlinelist():
    inlinelist = set()
    for f in FILENAME_INLINELIST:
        inlinelist.add(
            _module_dir(torch) + f.lstrip("torch.").replace(".", "/") + ".py"
        )
    return inlinelist


@functools.lru_cache(None)
def get_submodule_inlinelist():
    inlinelist = set()
    for f in SUBMODULE_INLINELIST:
        inlinelist.add(_module_dir(torch) + f.lstrip("torch.").replace(".", "/"))
    return inlinelist


if TYPE_CHECKING:
    for m in FILENAME_INLINELIST.union(SUBMODULE_INLINELIST):
        importlib.import_module(m)
    for f in FUNC_INLINELIST:
        eval(f)  # noqa: F841


# skip some standard python builtin libs
SKIP_DIRS = [
    "<frozen importlib",
    "<__array_function__ internals>",
] + [_module_dir(m) for m in BUILTIN_SKIPLIST]

SKIP_DIRS_RE = None

is_fbcode = importlib.import_module("torch._inductor.config").is_fbcode()
# Skip fbcode paths(including torch.package paths) containing
# one of the following strings.
FBCODE_SKIP_DIRS = {
    "torchrec/distributed",
    "torchrec/fb/distributed",
    "caffe2/torch/fb/sparsenn/pooled_embeddings_modules.py",
}
FBCODE_SKIP_DIRS_RE = re.compile(f".*({'|'.join(map(re.escape, FBCODE_SKIP_DIRS))})")


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


@dataclasses.dataclass
class SkipResult:
    skipped: bool
    reason: Optional[str]


# TODO(ybliang): This is a temp function, we should consolidate this with check_verbose.
def _check_verbose_inner(filename, allow_torch=False):
    """Should skip this file?"""
    if filename is None:
        return SkipResult(True, "filename is None")
    if filename in get_filename_inlinelist():
        return SkipResult(
            False,
            "inlined according skipfiles.FILENAME_INLINELIST",
        )
    # TODO(ybliang): the is_torch check should be consolidate with is_torch_inline_allowed
    if allow_torch and is_torch(filename):
        return SkipResult(
            False,
            "inlined according skipfiles.is_torch",
        )
    if is_fbcode and bool(FBCODE_SKIP_DIRS_RE.match(filename)):
        return SkipResult(
            True,
            "skipped according skipfiles.FBCODE_SKIP_DIRS",
        )
    if bool(SKIP_DIRS_RE.match(filename)):
        return SkipResult(True, "skipped according skipfiles.SKIP_DIRS")
    else:
        return SkipResult(False, "inlined by default")


def check_file(filename, allow_torch=False, extra_check=False):
    result = _check_verbose_inner(filename, allow_torch)
    if extra_check and result.skipped and is_torch_inline_allowed(filename):
        return SkipResult(
            False,
            "inlined according skipfiles.is_torch_inline_allowed returning True",
        )
    else:
        return result


def check_func(func, allow_torch=False, extra_check=False):
    if isinstance(
        func, (UserFunctionVariable, UserMethodVariable, NestedUserFunctionVariable)
    ):
        filename = func.get_filename()
        try:
            func = func.get_function()
        except NotImplementedError:
            func = None
    else:
        filename = getfile(func)
    if func in get_func_inlinelist():
        return SkipResult(
            False,
            "inlined according skipfiles.FUNC_INLINELIST",
        )
    return check_file(filename, allow_torch, extra_check)


# skip common third party libs
for _name in THIRDPARTY_SKIPLIST:
    add(_name)

_recompile_re()


def is_torch_inline_allowed(filename):
    if torch.distributed.is_available():
        from torch.distributed import _functional_collectives

        SUBMODULE_INLINELIST.add(_functional_collectives)

    return any(filename.startswith(_module_dir(mod)) for mod in SUBMODULE_INLINELIST)


@functools.lru_cache(None)
def dynamo_dir():
    import torch._dynamo

    return _module_dir(torch._dynamo)


def is_torch(filename):
    if filename.startswith(dynamo_dir()):
        return False
    return filename.startswith(_module_dir(torch))

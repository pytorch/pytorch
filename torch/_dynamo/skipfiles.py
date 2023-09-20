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
import glob
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
from typing import Optional

import torch
import torch._inductor.test_operators
import torch.distributed
import torch.utils._content_store

from . import comptime, external_utils

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

Dynamo skip/inline rules are defined as follows:
* Inline is the default behavior and will be used unless explicitly skipped.
* Dynamo has two SKIPLIST: BUILTIN_SKIPLIST and THIRDPARTY_SKIPLIST.
    * BUILTIN_SKIPLIST contains builtin python modules, such as abc, collections, etc.
    * THIRDPARTY_SKIPLIST contains common third party libraries, such as numpy, pandas, etc.
* Functions in these two SKIPLISTs are always skipped, except when they are explicitly
    put into the two ALLOWLIST: FILENAME_ALLOWLIST and SUBMODULE_ALLOWLIST.
* PyTorch(torch) is in the BUILTIN_SKIPLIST by default, but there are many cases
    where we want inline the functions under torch namespace. We should add them
    into FILENAME_ALLOWLIST or SUBMODULE_ALLOWLIST to make dynamo inline those functions.
* If you call functions under skipped modules/files, Dynamo will wrap these functions
    as SkipFilesVariable. There are a few functions(e.g, collections.OrderedDict) that
    we have special handling at SkipFilesVariable.call_function.

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
    torch,  # torch/* is skipped by default unless specified in FILENAME_ALLOWLIST or SUBMODULE_ALLOWLIST
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


# Force inline functions in these files, even the files is in *_SKIPLIST.
FILENAME_ALLOWLIST = {
    torch.nn.Sequential.__init__.__code__.co_filename,
    torch.set_rng_state.__code__.co_filename,
    torch._inductor.test_operators.__file__,
    torch.utils._content_store.__file__,
    external_utils.__file__,
    comptime.__file__,
    torch.optim._functional.__file__,
    torch.utils._foreach_utils.__file__,
    _module_dir(torch) + "ao/quantization/pt2e/qat_utils.py",
    _module_dir(torch) + "ao/quantization/quantizer/xnnpack_quantizer.py",
    _module_dir(torch) + "ao/quantization/pt2e/representation/rewrite.py",
    _module_dir(torch) + "ao/quantization/pt2e/utils.py",
    _module_dir(torch) + "ao/quantization/pt2e/eval_utils.py",
    _module_dir(torch) + "_export/constraints.py",
    _module_dir(torch) + "_higher_order_ops/cond.py",
    _module_dir(torch) + "_functorch/apis.py",
    _module_dir(torch) + "_functorch/deprecated.py",
    _module_dir(torch) + "distributed/tensor/parallel/_utils.py",
    _module_dir(torch) + "distributed/tensor/parallel/style.py",
    _module_dir(torch) + "distributed/tensor/parallel/_data_parallel_utils.py",
    _module_dir(torch) + "distributed/_tensor/api.py",
    _module_dir(torch) + "distributed/_tensor/device_mesh.py",
    _module_dir(torch) + "distributed/_tensor/placement_types.py",
    _module_dir(torch) + "distributed/c10d_logger.py",
    _module_dir(torch) + "distributed/_functional_collectives.py",
    torch.distributions.normal.__file__,
    torch.distributions.independent.__file__,
    torch.distributions.utils.__file__,
    torch.utils._contextlib.__file__,
    torch.fx._pytree.__file__,
}

if torch.distributed.is_available():
    # Inline the checkpoint code from distributed
    import torch.distributed.algorithms._checkpoint.checkpoint_wrapper

    FILENAME_ALLOWLIST |= {
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.__file__
    }

# Include optimizer code for tracing
FILENAME_ALLOWLIST |= {
    inspect.getfile(obj)
    for obj in torch.optim.__dict__.values()
    if inspect.isclass(obj)
}

# TODO (zhxchen17) Make exportdb importable here.
FILENAME_ALLOWLIST |= set(
    glob.glob(_module_dir(torch) + "_export/db/examples/*.py"),
) | {
    _module_dir(torch) + "_export/wrappers.py",
}


# Force inline functions under these modules, even the modules is in *_SKIPLIST.
SUBMODULE_ALLOWLIST = {
    torch.nn,
    torch.distributions,
    torch.testing,
    torch.ao.nn,
    torch._refs,
    torch._prims,
    torch._decomp,
    torch.utils._contextlib,
    torch.utils._pytree,
    torch.fx._pytree,
    torch.sparse,
}

if torch.distributed.is_available():
    from torch.distributed import _functional_collectives

    SUBMODULE_ALLOWLIST.add(_functional_collectives)


# skip some standard python builtin libs
SKIP_DIRS = [
    "<frozen importlib",
    "<__array_function__ internals>",
] + [_module_dir(m) for m in BUILTIN_SKIPLIST]

FILENAME_ALLOWLIST = {
    torch.nn.Sequential.__init__.__code__.co_filename,
    torch.set_rng_state.__code__.co_filename,
    torch._inductor.test_operators.__file__,
    torch.utils._content_store.__file__,
    # These are dynamo files!
    external_utils.__file__,
    comptime.__file__,  # Want to inline these helpers
}

if torch.distributed.is_available():
    # Inline the checkpoint code from distributed
    import torch.distributed.algorithms._checkpoint.checkpoint_wrapper

    FILENAME_ALLOWLIST |= {
        torch.distributed.algorithms._checkpoint.checkpoint_wrapper.__file__
    }

# Include optimizer code for tracing
FILENAME_ALLOWLIST |= {
    inspect.getfile(obj)
    for obj in torch.optim.__dict__.values()
    if inspect.isclass(obj)
}
FILENAME_ALLOWLIST |= {torch.optim._functional.__file__}
FILENAME_ALLOWLIST |= {torch.utils._foreach_utils.__file__}

# Do trace through match and replace patterns used in PT2E QAT
# Note: These patterns are comprised of torch ops and for internal use only.
# They are exported to aten graphs before being passed to the FX subgraph rewriter.
# TODO: find a better way to express this path without having to import
# `torch.ao.quantization.pt2e`, which interferes with memory profiling
FILENAME_ALLOWLIST |= {
    _module_dir(torch) + "ao/quantization/pt2e/qat_utils.py",
    _module_dir(torch) + "ao/quantization/quantizer/xnnpack_quantizer.py",
    _module_dir(torch) + "ao/quantization/pt2e/representation/rewrite.py",
    _module_dir(torch) + "ao/quantization/pt2e/utils.py",
    _module_dir(torch) + "ao/quantization/pt2e/eval_utils.py",
}

FILENAME_ALLOWLIST |= {
    _module_dir(torch) + "_export/constraints.py",
}

FILENAME_ALLOWLIST |= {
    _module_dir(torch) + "_higher_order_ops/cond.py",
}

# TODO (zhxchen17) Make exportdb importable here.
FILENAME_ALLOWLIST |= set(
    glob.glob(_module_dir(torch) + "_export/db/examples/*.py"),
) | {
    _module_dir(torch) + "_export/wrappers.py",
}

# torch.func: need to allow this file to be able to look at functorch transforms
FILENAME_ALLOWLIST |= {
    _module_dir(torch) + "_functorch/apis.py",
    _module_dir(torch) + "_functorch/deprecated.py",
}

FILENAME_ALLOWLIST |= {
    _module_dir(torch) + "distributed/tensor/parallel/_utils.py",
    _module_dir(torch) + "distributed/tensor/parallel/style.py",
    _module_dir(torch) + "distributed/tensor/parallel/_data_parallel_utils.py",
    _module_dir(torch) + "distributed/_tensor/api.py",
    _module_dir(torch) + "distributed/_tensor/device_mesh.py",
}

FILENAME_ALLOWLIST |= {
    _module_dir(torch) + "_dynamo/_trace_wrapped_higher_order_op.py",
}

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


def check(filename, allow_torch=False):
    """Should skip this file?"""
    if filename is None:
        return SkipResult(True, "filename is None")
    if filename in FILENAME_ALLOWLIST:
        return SkipResult(
            False,
            "allowlisted in skipfiles.FILENAME_ALLOWLIST",
        )
    if allow_torch and is_torch_inline_allowed(filename):
        return SkipResult(
            False,
            "allowlisted in skipfiles.SUBMODULE_ALLOWLIST",
        )
    if is_fbcode and bool(FBCODE_SKIP_DIRS_RE.match(filename)):
        return SkipResult(
            True,
            "should be skipped according skipfiles.FBCODE_SKIP_DIRS",
        )
    if bool(SKIP_DIRS_RE.match(filename)):
        return SkipResult(True, "should be skipped according skipfiles.SKIP_DIRS")
    else:
        return SkipResult(False, "inlining by default")


# skip common third party libs
for _name in THIRDPARTY_SKIPLIST:
    add(_name)

_recompile_re()


def is_torch_inline_allowed(filename):
    return any(filename.startswith(_module_dir(mod)) for mod in SUBMODULE_ALLOWLIST)


@functools.lru_cache(None)
def dynamo_dir():
    import torch._dynamo

    return _module_dir(torch._dynamo)


def is_torch(filename):
    if filename.startswith(dynamo_dir()):
        return False
    return filename.startswith(_module_dir(torch))

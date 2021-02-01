"""Allows benchmark utils to work with other versions of PyTorch

Motivation:
  There are two workflows where benchmark utils could be used. One is the
  obvious case where someone simply builds or installs PyTorch and uses Timer.
  The other is that the entire `torch/utils/benchmark` folder from a CURRENT
  PyTorch checkout is copy-pasted into a much OLDER version of the PyTorch
  source code. This is what we refer to here as "back testing". The rationale
  is that we might want to use current tooling to study some aspect of an
  earlier version of PyTorch. (e.g. a regression.)

  The problem is that Timer relies on several aspects of core PyTorch, namely
  some binding functions for Valgrind symbols in `torch._C` and the
  `torch.__config__._cxx_flags()` method. If we were to naively copy code
  around this wouldn't work as the symbols of interest aren't present in
  earlier versions of PyTorch. In order to work around this, we must add back
  testing shims. These shims will never activate during normal use, but will
  allow Timer to function outside of the "correct" version of PyTorch by
  emulating functionality that was added later.

  These shims are temporary, and as Timer becomes more integrated with
  PyTorch the cost and complexity of such shims will increase. Once back
  testing is no longer required (which is to say we have done enough historic
  analysis and the shims no longer justify their maintenance and code
  complexity costs) back testing paths will be removed.
"""
from typing import TYPE_CHECKING

import torch


IS_BACK_TESTING_OVERRIDE: bool = False  # patch will set to True.
IS_BACK_TESTING: bool = not all([
    hasattr(torch._C, "_valgrind_supported_platform"),
    hasattr(torch._C, "_valgrind_toggle"),
    hasattr(torch._C, "_valgrind_dump_stats"),
    hasattr(torch, "__config__"),
    hasattr(torch.__config__, "_cxx_flags")
]) or IS_BACK_TESTING_OVERRIDE


if TYPE_CHECKING:
    # We only care about type checking the current version.
    IS_BACK_TESTING = False


if IS_BACK_TESTING:
    CXX_FLAGS = ["-O2", "-fPIC", "-g"]
    ScriptFunction = (
        getattr(torch.jit, "ScriptFunction", None) or
        getattr(torch._C, "Function", None) or
        getattr(torch._C, "GraphExecutor", None)
    )
else:
    CXX_FLAGS = torch.__config__._cxx_flags().strip().split()
    if "-g" not in CXX_FLAGS:
        CXX_FLAGS.append("-g")
    ScriptFunction = torch.jit.ScriptFunction

# Owner(s): ["module: dsl-native-ops"]
"""Run test/test_sort_and_select.py with every CUDA sort through narrow out= indices.

The override serves only narrow out= indices, which those tests never pass. A
TorchFunctionMode rewrites every functional CUDA sort/argsort (including the ones
topk/unique tests use as references) into the out-variant with the narrowest index
dtype that holds the row, and widens the indices back to int64 for the test's own
assertions. The cond's short-call gate is off, so bool keys and short single rows
still reach their kernels; calls the cond accepts with the gate off must launch
the kernel its router picks, and the rest (0-dim, empty, unsupported key dtypes)
check the narrow out= fallback.
"""

import functools
import os
import sys
from unittest import mock

import torch
from torch.overrides import TorchFunctionMode
from torch.testing._internal.common_utils import run_tests


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_sort_and_select

from torch._native.ops.sort import (
    cutedsl_global,
    cutedsl_impl,
    cutedsl_inplace,
    cutedsl_partition,
    cutedsl_register,
)


_NARROW = (
    (torch.uint8, 256),
    (torch.uint16, 65536),
    (torch.int32, 2**31),
    (torch.uint32, 2**32),
)
_SORTS = {torch.sort, torch.Tensor.sort}
_ARGSORTS = {torch.argsort, torch.Tensor.argsort}


class _NarrowOutMode(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.launched = []
        self.rewritten = 0
        self.missed = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if (func not in _SORTS and func not in _ARGSORTS) or "out" in kwargs:
            return func(*args, **kwargs)
        x, *rest = args
        names = (
            ("dim", "descending") if func in _SORTS else ("dim", "descending", "stable")
        )
        call = dict(zip(names, rest)) | kwargs
        if not isinstance(x, torch.Tensor) or not x.is_cuda:
            return func(*args, **kwargs)
        dim = call.get("dim", -1)
        if isinstance(dim, str):
            dim = x.names.index(dim)
        nsort = x.size(dim) if x.dim() else 1
        index_dtype = next((dtype for dtype, cap in _NARROW if nsort <= cap), None)
        if index_dtype is None:
            return func(*args, **kwargs)
        # Functional sort allocates its outputs like self; keep that layout.
        values = torch.empty_like(x)
        indices = torch.empty_like(x, dtype=index_dtype)
        stable = call.get("stable")
        descending = bool(call.get("descending", False))
        expected = None
        # Judged with the gate off, so dropping setUp's patch fails the calls
        # the gate would send to stock.
        with mock.patch.object(cutedsl_impl, "jit_pays_off", lambda *a: True):
            accepted = cutedsl_impl._out_cond(
                x,
                stable=stable,
                dim=dim,
                descending=descending,
                values=values,
                indices=indices,
            )
        if accepted:
            launcher = cutedsl_impl._launcher(x, nsort, bool(stable), descending)
            expected = launcher.func.__name__
        self.rewritten += 1
        start = len(self.launched)
        if func in _ARGSORTS:
            # Only the stable= overload of argsort has an out variant.
            torch.argsort(x, **(call | {"stable": bool(stable)}), out=indices)
            result = indices.long()
        else:
            torch.sort(x, **call, out=(values, indices))
            result = torch.return_types.sort((values, indices.long()))
        if expected is not None and expected not in self.launched[start:]:
            self.missed.append((tuple(x.shape), x.dtype, call, expected))
        return result


_KERNELS = (
    (cutedsl_inplace, "inplace_sort"),
    (cutedsl_global, "global_sort"),
    (cutedsl_register, "register_sort"),
    (cutedsl_partition, "partition_sort"),
)


def _narrow_out(cls):
    class NarrowOut(cls):
        def setUp(self):
            super().setUp()
            self.mode = _NarrowOutMode()

            def counting(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    self.mode.launched.append(fn.__name__)
                    return fn(*args, **kwargs)

                return wrapper

            # AOT kernels would serve covered calls invisibly to these counters.
            aot_enabled = torch._C._get_native_aot_enabled()
            torch._native.set_aot_enabled(False)
            self.addCleanup(torch._native.set_aot_enabled, aot_enabled)
            # The tests' tensors are small; the gate would send most to stock.
            patcher = mock.patch.object(cutedsl_impl, "jit_pays_off", lambda *a: True)
            patcher.start()
            self.addCleanup(patcher.stop)
            for module, name in _KERNELS:
                patcher = mock.patch.object(
                    module, name, counting(getattr(module, name))
                )
                patcher.start()
                self.addCleanup(patcher.stop)
            self.mode.__enter__()

        def tearDown(self):
            # Before super(), which fails on a leaked mode.
            self.mode.__exit__(None, None, None)
            super().tearDown()
            self.assertEqual(
                self.mode.missed, [], "rewritten calls that missed the routed kernel"
            )

    NarrowOut.__name__ = NarrowOut.__qualname__ = f"{cls.__name__}NarrowOut"
    return NarrowOut


if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
    for _cls in (
        test_sort_and_select.TestSortAndSelectDeviceCUDA,
        test_sort_and_select.TestSortAndSelectCUDACUDA,
    ):
        _narrow = _narrow_out(_cls)
        globals()[_narrow.__name__] = _narrow


if __name__ == "__main__":
    run_tests()

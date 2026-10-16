# Owner(s): ["module: dsl-native-ops"]

import sys
import unittest
from collections import Counter

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

from torch._native import registry as R


_OVERRIDES_MODULE = "torch._native.ops.reductions.overrides"

DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "f8e4m3": torch.float8_e4m3fn,
    "f8e4m3z": torch.float8_e4m3fnuz,
    "f8e5m2": torch.float8_e5m2,
    "f8e5m2z": torch.float8_e5m2fnuz,
    "f8e8m0": torch.float8_e8m0fnu,
    "c64": torch.complex64,
    "c128": torch.complex128,
    "i32": torch.int32,
    "i64": torch.int64,
    "i8": torch.int8,
    "i16": torch.int16,
    "u8": torch.uint8,
    "u16": torch.uint16,
    "u32": torch.uint32,
    "u64": torch.uint64,
    "bool": torch.bool,
}

OPS = {
    "sum": lambda t, **k: torch.sum(t, **k),
    "mean": lambda t, **k: torch.mean(t, **k),
    "nansum": lambda t, **k: torch.nansum(t, **k),
    "prod": lambda t, **k: (
        torch.prod(t, k["dim"], k.get("keepdim", False))
        if "dim" in k
        else torch.prod(t)
    ),
    "amax": lambda t, **k: torch.amax(t, **k),
    "amin": lambda t, **k: torch.amin(t, **k),
    "argmax": lambda t, **k: torch.argmax(t, **k),
    "max.dim": lambda t, **k: torch.max(t, **k) if "dim" in k else torch.max(t),
    "min.dim": lambda t, **k: torch.min(t, **k) if "dim" in k else torch.min(t),
    "aminmax": lambda t, **k: torch.aminmax(t, **k),
    "var": lambda t, **k: torch.var(t, **k),
    "std": lambda t, **k: torch.std(t, **k),
    "var_mean": lambda t, **k: torch.var_mean(t, **k),
    "std_mean": lambda t, **k: torch.std_mean(t, **k),
    "vector_norm": lambda t, **k: torch.linalg.vector_norm(t, **k),
    "all": lambda t, **k: torch.all(t, **k),
    "any": lambda t, **k: torch.any(t, **k),
    "count_nonzero": lambda t, **k: torch.count_nonzero(t, **k),
}

BASE_OPTION = ("dim=-1", {"dim": -1}, {})
OPTIONS = [
    ("dim=0", {"dim": 0}, {}),
    ("dim=(0,1)", {"dim": (0, 1)}, {}),
    ("keepdim", {"dim": -1, "keepdim": True}, {}),
    ("full", {}, {}),
    ("noncontig", {"dim": -1}, {"contiguous": False}),
    ("empty-output", {"dim": 1}, {"empty": "output"}),
    ("empty-axis", {"dim": 1}, {"empty": "axis"}),
    ("0-dim", {"dim": 0}, {"zerodim": True}),
]

EXPECTED_DECLINES = set()


def _build(dtype, contiguous=True, empty=None, zerodim=False):
    if zerodim:
        return torch.zeros((), device="cuda").to(dtype)
    shape = {"output": (0, 5), "axis": (5, 0)}.get(empty, (64, 128))
    if dtype.is_complex:
        real_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
        real = torch.randn(shape, device="cuda", dtype=real_dtype)
        imag = torch.randn(shape, device="cuda", dtype=real_dtype)
        t = torch.complex(real, imag)
    elif dtype.is_floating_point:
        t = torch.randn(shape, device="cuda", dtype=torch.float32).to(dtype)
    else:
        t = torch.randint(0, 7, shape, device="cuda").to(dtype)
    return t if contiguous else t.transpose(0, 1)


def _cells():
    for op_name, fn in OPS.items():
        for dt_name, dtype in DTYPES.items():
            yield op_name, fn, dt_name, dtype, BASE_OPTION
        for option in OPTIONS:
            yield op_name, fn, "f32", torch.float32, option


def _served(fn):
    """Run `fn` and report whether a reduction condition accepted it."""

    def is_ours(node):
        return getattr(node.cond_fn, "__module__", "") == _OVERRIDES_MODULE

    accepted = [False]

    def wrap(f):
        def probe(*args, **kwargs):
            r = f(*args, **kwargs)
            if r:
                accepted[0] = True
            return r

        return probe

    ours = {k: v for k, v in R._graphs.items() if any(is_ours(n) for n in v)}
    saved = [(n, n.cond_fn) for nodes in ours.values() for n in nodes if is_ours(n)]
    for n, _ in saved:
        n.cond_fn = wrap(n.cond_fn)
    for (lib_symbol, op_symbol, key), nodes in ours.items():
        R._register_overrides_from_graph(lib_symbol, op_symbol, key, nodes)
    try:
        out = fn()
    finally:
        for n, orig in saved:
            n.cond_fn = orig
        for (lib_symbol, op_symbol, key), nodes in ours.items():
            R._register_overrides_from_graph(lib_symbol, op_symbol, key, nodes)
    return accepted[0], out


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestReductionCoverage(TestCase):
    def _agree(self, got, ref):
        pairs = zip(
            got if isinstance(got, tuple) else (got,),
            ref if isinstance(ref, tuple) else (ref,),
        )
        for g, r in pairs:
            if g.dtype != r.dtype:
                return False
            if g.is_floating_point() or g.is_complex():
                if not torch.allclose(g, r, rtol=1e-2, atol=1e-2, equal_nan=True):
                    return False
            elif not torch.equal(g, r):
                return False
        return True

    def test_every_declared_cell_is_served(self):
        holes, wrong, counts = [], [], Counter()
        for op_name, fn, dt_name, dtype, option in _cells():
            label, kwargs, build_kw = option
            cell = (op_name, dt_name, label)
            t = _build(dtype, **build_kw)
            call = lambda t=t, fn=fn, kw=kwargs: fn(t, **kw)  # noqa: E731
            try:
                with torch.backends.python_native.cutedsl.disabled():
                    ref = call()
            except Exception:
                counts["aten-rejects"] += 1
                continue
            ok, got = _served(call)
            if not self._agree(got, ref):
                wrong.append(cell)
            elif ok:
                counts["served"] += 1
            else:
                counts["declined"] += 1
                if cell not in EXPECTED_DECLINES:
                    holes.append(cell)

        self.assertEqual(
            wrong, [], f"cells where our answer differs from ATen: {wrong}"
        )
        self.assertEqual(holes, [], f"cells silently declined to ATen: {holes}")
        self.assertEqual(sum(counts.values()), sum(1 for _ in _cells()))
        self.assertGreater(counts["served"], 0)
        self.assertEqual(counts["declined"], len(EXPECTED_DECLINES))


if __name__ == "__main__":
    run_tests()

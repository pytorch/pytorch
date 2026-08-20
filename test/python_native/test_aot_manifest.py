# Owner(s): ["module: dsl-native-ops"]
"""Tests for the native-AOT runtime layer in torch/_native:

  * aot_manifest.covers() / get_coverage(): declaration-driven
    JIT-coverage checks (the router consults coverage once per call)
  * the native-AOT Context switch (set_aot_enabled/aot_enabled) and
    its integration with python_native.<dsl>.disabled()
  * _native_aot_embedded() detection and the env opt-out

None of these need the AOT kernel library or a GPU; tensors are CPU
where a tensor is needed at all.
"""

import os
import subprocess
import sys
import tempfile
import unittest.mock as mock

import torch
from torch._native import aot_manifest
from torch.testing._internal.common_utils import run_tests, TestCase


# A minimal declaration module (see tools/native_aot/decl.py for the
# contract). covered_axes() runs as ordinary Python, so argument
# defaults, dim normalization etc. are plain code.
DECLARATION = """\
ATEN_OP = "fakeop"
DISPATCH_KEY = "CUDA"
KERNEL_MODULE = "kernel.py"


def kernel_precompile_grid():
    return [
        {"dtype": ["float32", "bfloat16"], "N": [1024, 2048], "K": [8, 16],
         "deterministic": False},
    ]


def covered_axes(self, k, dim=-1, largest=True, sorted=True):
    import torch

    return {
        "dtype": self.dtype,
        "N": self.shape[-1] if self.dim() >= 1 else 0,
        "K": k,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def cpp_dispatch(spec):
    return "true"


def cpp_launch(spec, launch_fn):
    return f"{launch_fn}();"
"""


class ManifestFixture:
    """Context manager: a temp ops dir with one fakeop declaration,
    patched into aot_manifest with its cache cleared on entry/exit."""

    def __init__(self, body: str = DECLARATION, op: str = "fakeop"):
        self.body, self.op = body, op

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        op_dir = os.path.join(self._dir.name, self.op)
        os.makedirs(op_dir)
        with open(os.path.join(op_dir, "aot.py"), "w") as f:
            f.write(self.body)
        self._patch = mock.patch.object(aot_manifest, "_OPS_DIR", self._dir.name)
        self._patch.start()
        aot_manifest._load_coverage.cache_clear()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        aot_manifest._load_coverage.cache_clear()
        self._dir.cleanup()


class TestCovers(TestCase):
    def _covered_tensor(self):
        return torch.empty(4, 1024, dtype=torch.float32)

    def test_covered_call(self):
        with ManifestFixture():
            self.assertTrue(
                aot_manifest.covers("fakeop", "CUDA", (self._covered_tensor(), 8), {})
            )

    def test_defaults_fill_omitted_args(self):
        # k-only call: dim/largest/sorted come from covered_axes' own
        # signature defaults.
        with ManifestFixture():
            x = self._covered_tensor()
            self.assertTrue(aot_manifest.covers("fakeop", "CUDA", (x, 8), {}))
            self.assertTrue(aot_manifest.covers("fakeop", "CUDA", (x,), {"k": 8}))

    def test_off_grid_values_uncovered(self):
        with ManifestFixture():
            x = self._covered_tensor()
            self.assertFalse(
                aot_manifest.covers("fakeop", "CUDA", (x, 32), {})
            )  # K off-grid
            self.assertFalse(
                aot_manifest.covers("fakeop", "CUDA", (torch.empty(4, 512), 8), {})
            )  # N off-grid
            xh = torch.empty(4, 1024, dtype=torch.float16)
            self.assertFalse(
                aot_manifest.covers("fakeop", "CUDA", (xh, 8), {})
            )  # dtype off-grid

    def test_bf16_on_grid(self):
        with ManifestFixture():
            xb = torch.empty(4, 2048, dtype=torch.bfloat16)
            self.assertTrue(aot_manifest.covers("fakeop", "CUDA", (xb, 16), {}))

    def test_runtime_state_bind(self):
        with ManifestFixture():
            x = self._covered_tensor()
            prior = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                self.assertFalse(aot_manifest.covers("fakeop", "CUDA", (x, 8), {}))
            finally:
                torch.use_deterministic_algorithms(prior)
            self.assertTrue(aot_manifest.covers("fakeop", "CUDA", (x, 8), {}))

    def test_overload_shares_base_manifest(self):
        with ManifestFixture():
            self.assertTrue(
                aot_manifest.covers(
                    "fakeop.values", "CUDA", (self._covered_tensor(), 8), {}
                )
            )

    def test_unknown_op_or_key_uncovered(self):
        with ManifestFixture():
            x = self._covered_tensor()
            self.assertFalse(aot_manifest.covers("otherop", "CUDA", (x, 8), {}))
            self.assertFalse(aot_manifest.covers("fakeop", "CPU", (x, 8), {}))

    def test_bind_failure_is_uncovered(self):
        # A covered_axes call that raises (attribute missing on the
        # argument, or arguments that don't match its signature) must
        # degrade to "uncovered", not propagate.
        with ManifestFixture():
            self.assertFalse(aot_manifest.covers("fakeop", "CUDA", (object(), 8), {}))
            # Too few args to bind at all:
            self.assertFalse(aot_manifest.covers("fakeop", "CUDA", (), {}))


class TestGetCoverage(TestCase):
    def test_declared_op_has_coverage(self):
        with ManifestFixture():
            c = aot_manifest.get_coverage("fakeop", "CUDA")
            self.assertIsNotNone(c)
            self.assertTrue(c.covers((torch.empty(4, 1024), 8), {}))
            self.assertFalse(c.covers((torch.empty(4, 512), 8), {}))

    def test_undeclared_op_returns_none(self):
        with ManifestFixture():
            self.assertIsNone(aot_manifest.get_coverage("otherop", "CUDA"))
            self.assertIsNone(aot_manifest.get_coverage("fakeop", "CPU"))

    def test_variant_symbols_share_base_coverage(self):
        # Overload-qualified and in-place symbols funnel through the same
        # structured wrapper, so they must resolve to the base declaration.
        with ManifestFixture():
            base = aot_manifest.get_coverage("fakeop", "CUDA")
            self.assertIs(aot_manifest.get_coverage("fakeop.values", "CUDA"), base)
            self.assertIs(aot_manifest.get_coverage("fakeop_", "CUDA"), base)

    def test_router_declines_covered_calls_once(self):
        # The router consults coverage ONCE per call ahead of the cond
        # chain: with two registered override paths, a covered call must
        # run covered_axes exactly once and no cond at all; an uncovered
        # call runs the conds.
        from torch._native import registry

        with ManifestFixture():
            cond_calls, axes_calls = [], []
            coverage = aot_manifest.get_coverage("fakeop", "CUDA")
            inner_axes = coverage._covered_axes

            def counting_axes(*args, **kwargs):
                axes_calls.append(1)
                return inner_axes(*args, **kwargs)

            coverage._covered_axes = counting_axes

            def probe_cond(*args, **kwargs):
                cond_calls.append(1)
                return False

            cond_impl = [(probe_cond, "a"), (probe_cond, "b")]
            _NO_MATCH = object()

            # Mirror of _register_overrides_from_graph._dispatch; the
            # closure itself is not importable, so keep this in sync.
            def dispatch(args, kwargs):
                if coverage is not None and coverage.covers(args, kwargs):
                    return _NO_MATCH
                for cond, _impl in cond_impl:
                    if cond(*args, **kwargs):
                        return None
                return _NO_MATCH

            self.assertIs(dispatch((torch.empty(4, 1024), 8), {}), _NO_MATCH)
            self.assertEqual(len(axes_calls), 1)
            self.assertEqual(cond_calls, [])
            dispatch((torch.empty(4, 512), 8), {})
            self.assertEqual(len(cond_calls), 2)

        # And the real router build must reference get_coverage: guard
        # against the check being dropped from registry._dispatch.
        import inspect

        src = inspect.getsource(registry._register_overrides_from_graph)
        self.assertIn("get_coverage", src)
        self.assertIn("coverage.covers(args, kwargs)", src)


class TestAotContextSwitch(TestCase):
    def test_masked_switch_restores_jit_coverage(self):
        # With AOT masked, coverage must report uncovered so covered
        # calls keep their JIT route (declining into a stub that will
        # not fire would silently lose BOTH accelerated routes).
        from torch import _native

        with ManifestFixture():
            x = torch.empty(4, 1024, dtype=torch.float32)
            self.assertTrue(aot_manifest.covers("fakeop", "CUDA", (x, 8), {}))
            try:
                _native.set_aot_enabled(False)
                self.assertFalse(aot_manifest.covers("fakeop", "CUDA", (x, 8), {}))
            finally:
                _native.set_aot_enabled(True)
            self.assertTrue(aot_manifest.covers("fakeop", "CUDA", (x, 8), {}))

    def test_set_get_roundtrip(self):
        from torch import _native

        self.assertTrue(_native.aot_enabled())
        try:
            _native.set_aot_enabled(False)
            self.assertFalse(_native.aot_enabled())
        finally:
            _native.set_aot_enabled(True)
        self.assertTrue(_native.aot_enabled())

    def test_disabled_context_masks_and_restores(self):
        from torch import _native

        pn = torch.backends.python_native
        self.assertTrue(_native.aot_enabled())
        with pn.cutedsl.disabled():
            self.assertFalse(_native.aot_enabled())
        self.assertTrue(_native.aot_enabled())

    def test_disabled_context_restores_on_exception(self):
        from torch import _native

        pn = torch.backends.python_native
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with pn.cutedsl.disabled():
                raise RuntimeError("boom")
        self.assertTrue(_native.aot_enabled())


class TestEmbedDetection(TestCase):
    def test_disable_env_var_flips_context_switch(self):
        # Subprocess: on an embedded build the opt-out masks the kernels
        # by flipping the Context switch; gated coverage then declines,
        # so covered calls keep their JIT route. On a build without
        # artifacts the switch stays on (nothing to mask).
        code = (
            "import torch\n"
            "from torch._native import _native_aot_embedded, aot_enabled\n"
            "if _native_aot_embedded():\n"
            "    assert not aot_enabled()\n"
            "else:\n"
            "    assert aot_enabled()\n"
            "print('OPT_OUT_OK')\n"
        )
        env = dict(os.environ, TORCH_DISABLE_NATIVE_AOT="1")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OPT_OUT_OK", proc.stdout)

    def test_import_does_not_initialize_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        code = (
            "import torch\nassert not torch.cuda.is_initialized()\nprint('LAZY_OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("LAZY_OK", proc.stdout)


if __name__ == "__main__":
    run_tests()

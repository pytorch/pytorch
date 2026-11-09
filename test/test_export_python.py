# Owner(s): ["oncall: pt2"]
import ast
import functools
import importlib
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
import uuid

import torch
from torch.compiler import PrecompileError
from torch.compiler._export_python import _CODE_BEGIN, _CODE_END
from torch.compiler.precompile import no_compilation
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


# export_python serves an artifact by importing the module its function was captured
# from, so each test writes its functions into a fresh importable module.
_MODULE = """
import os

import torch

def scaled(x, *, scale=2.0):
    return (x * scale).relu()

def breaks(x):
    y = x.sin()
    print("graph break")
    return y.cos()

def matmul(x, y):
    return x @ y

@torch.compile(backend="eager")
def compiled_double(x):
    return x * 2

def calls_compiled(x):
    return compiled_double(x) + 1

def bump(x):
    x.mul_(2)
    return x + 1

def grad_aware(x):
    return x * 2 if x.requires_grad else x * 3

_ARTIFACTS = os.path.join(os.path.dirname(__file__), "decorated")

@torch.compiler.export_python(path=os.path.join(_ARTIFACTS, "scaled.py"))
def decorated_scaled(x):
    return (x * 2.0).relu()

def shifted(x):
    return x * 7.25 + 1

def reduce(x):
    return (x.sin() * 2).sum(dim=-1)

def linear_relu(x, w):
    return (x @ w).relu()

from torch.nn.attention.flex_attention import flex_attention

def causal(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))

def attention(q, k, v):
    return flex_attention(q, k, v, score_mod=causal)

"""


def _graph_code(path):
    with open(path) as f:
        return f.read().partition(_CODE_BEGIN)[2].rpartition(_CODE_END)[0]


def _edit_graph_code(path, old, new):
    # Only the graph code, as an agent hill-climbing the file would.
    with open(path) as f:
        head, begin, rest = f.read().partition(_CODE_BEGIN)
    code, end, tail = rest.rpartition(_CODE_END)
    if old not in code:
        raise AssertionError(f"{old!r} not in the graph code of {path}")
    with open(path, "w") as f:
        f.write(head + begin + code.replace(old, new) + end + tail)


def _in_a_fresh_process(test, body):
    if TEST_WITH_CROSSREF:
        # The crossref harness keeps a dispatch mode active while the capture runs,
        # and Dynamo guards on it; a fresh process has no such mode, so every call
        # would miss there.
        test.skipTest("a fresh process lacks the crossref dispatch mode")
    script = (
        f"import sys, torch\nsys.path.insert(0, {test.dir!r})\n"
        f"import {test.module_name} as m\n{textwrap.dedent(body)}\nprint('served')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=900
    )
    test.assertEqual(out.returncode, 0, out.stderr)
    test.assertIn("served", out.stdout)
    return out.stdout


class _ExportPythonTestCase(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.module_name = f"export_python_fns_{uuid.uuid4().hex}"
        with open(os.path.join(self.dir, self.module_name + ".py"), "w") as f:
            f.write(_MODULE)
        sys.path.insert(0, self.dir)
        self.addCleanup(sys.path.remove, self.dir)
        self.module = importlib.import_module(self.module_name)
        self.addCleanup(sys.modules.pop, self.module_name, None)

    def _export(self, name, path, **kwargs):
        return torch.compiler.export_python(path=path, **kwargs)(
            getattr(self.module, name)
        )


@skipIfTorchDynamo("precompile captures cannot run under dynamo wrapping")
class TestExportPython(_ExportPythonTestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_first_call_captures_and_later_calls_serve_the_file(self):
        path = os.path.join(self.dir, "artifacts", "scaled.py")
        run = self._export("scaled", path)
        x = torch.randn(8)
        self.assertEqual(run(x, scale=3.0), (x * 3.0).relu())
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(path + ".cache"))
        with no_compilation():
            self.assertEqual(run(x, scale=3.0), (x * 3.0).relu())
            y = torch.randn(8)
            self.assertEqual(run(y, scale=3.0), (y * 3.0).relu())

    def test_the_file_is_the_graph_code_and_one_captured_line(self):
        path = os.path.join(self.dir, "scaled.py")
        self._export("scaled", path)(torch.randn(8))
        with open(path) as f:
            lines = f.read().splitlines()
        self.assertTrue(
            lines[0].startswith("# Exported by torch.compiler.export_python")
        )
        self.assertIn("def call(", _graph_code(path))
        self.assertTrue(lines[-1].startswith("_CAPTURED = '"))
        # Nothing between the graph code and that line but a comment.
        tail = lines[lines.index(_CODE_END) + 1 : -1]
        self.assertTrue(all(not line or line.startswith("#") for line in tail), tail)

    def test_a_graph_break_is_refused(self):
        path = os.path.join(self.dir, "breaks.py")
        with self.assertRaisesRegex(PrecompileError, "does not trace as one graph"):
            self._export("breaks", path)(torch.randn(4))
        self.assertFalse(os.path.exists(path))

    def test_a_call_that_needs_a_backward_graph_is_refused(self):
        path = os.path.join(self.dir, "scaled.py")
        run = self._export("scaled", path)
        x = torch.randn(8, requires_grad=True)
        with self.assertRaisesRegex(PrecompileError, "backward graph too"):
            run(x)
        self.assertFalse(os.path.exists(path))
        with torch.no_grad():
            self.assertEqual(run(x), (x * 2.0).relu())
            with no_compilation():
                self.assertEqual(run(x), (x * 2.0).relu())

    def test_a_new_process_serves_the_file_without_compiling(self):
        path = os.path.join(self.dir, "scaled.py")
        self._export("scaled", path)(torch.randn(4))
        _in_a_fresh_process(
            self,
            f"""
            from torch.compiler.precompile import no_compilation
            run = torch.compiler.export_python(path={path!r})(m.scaled)
            x = torch.randn(4)
            with no_compilation():
                torch.testing.assert_close(run(x), (x * 2.0).relu())
            """,
        )

    def test_the_decorator_serves_here_and_in_a_new_process(self):
        # The module binds export_python's wrapper under the decorated name.
        x = torch.randn(4)
        self.assertEqual(self.module.decorated_scaled(x), (x * 2.0).relu())
        with no_compilation():
            self.assertEqual(self.module.decorated_scaled(x), (x * 2.0).relu())
        _in_a_fresh_process(
            self,
            """
            from torch.compiler.precompile import no_compilation
            x = torch.randn(4)
            with no_compilation():
                torch.testing.assert_close(m.decorated_scaled(x), (x * 2.0).relu())
            """,
        )

    def test_a_first_call_that_raises_writes_nothing_and_the_next_call_captures(self):
        path = os.path.join(self.dir, "matmul.py")
        run = self._export("matmul", path)
        x, y = torch.randn(2, 3), torch.randn(3, 2)
        with self.assertRaises(RuntimeError):
            run(x, x)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(run(x, y), x @ y)
        with no_compilation():
            self.assertEqual(run(x, y), x @ y)

    def test_deleting_the_file_recaptures(self):
        path = os.path.join(self.dir, "scaled.py")
        x = torch.randn(8)
        self._export("scaled", path)(x)
        os.remove(path)
        run = self._export("scaled", path)
        self.assertEqual(run(x, scale=5.0), (x * 5.0).relu())
        self.assertTrue(os.path.exists(path))

    def test_an_uncaptured_call_raises_instead_of_compiling(self):
        path = os.path.join(self.dir, "scaled.py")
        run = self._export("scaled", path)
        run(torch.randn(8))
        with (
            no_compilation(),
            self.assertRaisesRegex(PrecompileError, "no captured variant"),
        ):
            run(torch.randn(3, 3))

    def test_an_edited_file_runs_the_edit(self):
        path = os.path.join(self.dir, "shifted.py")
        x = torch.randn(8)
        self.assertEqual(self._export("shifted", path)(x), x * 7.25 + 1)
        _edit_graph_code(path, "7.25", "3.5")
        with self.assertRaisesRegex(
            PrecompileError, "other code for the same compiled"
        ):
            self._export("shifted", path)(x)
        _in_a_fresh_process(
            self,
            f"""
            run = torch.compiler.export_python(path={path!r})(m.shifted)
            x = torch.ones(8)
            torch.testing.assert_close(run(x), x * 3.5 + 1)
            """,
        )

    def test_a_missing_cache_still_serves(self):
        path = os.path.join(self.dir, "shifted.py")
        x = torch.randn(8)
        self._export("shifted", path)(x)
        with self.assertNoLogs("torch.compiler._export_python", "INFO"):
            self.assertEqual(self._export("shifted", path)(x), x * 7.25 + 1)
        os.remove(path + ".cache")
        with self.assertLogs("torch.compiler._export_python", "INFO") as logs:
            self.assertEqual(self._export("shifted", path)(x), x * 7.25 + 1)
        self.assertIn("serving it without the cache", logs.output[0])
        with (
            no_compilation(),
            self.assertRaisesRegex(PrecompileError, "delete .* to recapture"),
        ):
            self._export("shifted", path)(x)

    def test_a_corrupt_cache_is_reported_as_corrupt(self):
        path = os.path.join(self.dir, "shifted.py")
        x = torch.randn(8)
        self._export("shifted", path)(x)
        with open(path + ".cache", "wb") as f:
            f.write(b"not a cache")
        with self.assertLogs("torch._precompile", "WARNING") as logs:
            self.assertEqual(self._export("shifted", path)(x), x * 7.25 + 1)
        self.assertIn("could not read the cache envelope", logs.output[0])
        os.remove(path + ".cache")
        os.mkdir(path + ".cache")
        with self.assertRaisesRegex(PrecompileError, "could not read the cache"):
            self._export("shifted", path)(x)

    def test_check_catches_an_edit_that_changes_the_numerics(self):
        path = os.path.join(self.dir, "shifted.py")
        x = torch.randn(8)
        self._export("shifted", path, check=True)(x)
        self.assertEqual(self._export("shifted", path, check=True)(x), x * 7.25 + 1)
        _edit_graph_code(path, "7.25", "7.5")
        _in_a_fresh_process(
            self,
            f"""
            checked = torch.compiler.export_python(path={path!r}, check=True)(m.shifted)
            try:
                checked(torch.ones(8))
            except torch.compiler.PrecompileError as e:
                if "does not match eager shifted" not in str(e):
                    raise
            else:
                raise AssertionError("the check did not catch the edit")
            """,
        )

    def test_check_runs_fn_eagerly_and_compares_in_place_writes(self):
        x = torch.randn(4)
        compiled = os.path.join(self.dir, "calls_compiled.py")
        bump = os.path.join(self.dir, "bump.py")
        self._export("calls_compiled", compiled)(x)
        self._export("bump", bump)(x.clone())
        _in_a_fresh_process(
            self,
            f"""
            from torch.compiler.precompile import no_compilation
            x = torch.randn(4)
            # The eager run of a torch.compile'd callee compiles nothing.
            run = torch.compiler.export_python(path={compiled!r}, check=True)
            with no_compilation():
                torch.testing.assert_close(run(m.calls_compiled)(x), x * 2 + 1)
            # The copy is taken before the artifact doubles y, so eager doubles it too.
            y = x.clone()
            out = torch.compiler.export_python(path={bump!r}, check=True)(m.bump)(y)
            torch.testing.assert_close((out, y), (x * 2 + 1, x * 2))
            """,
        )

    def test_check_copies_keep_requires_grad(self):
        path = os.path.join(self.dir, "grad_aware.py")
        x = torch.randn(4, requires_grad=True)
        with torch.no_grad():
            self._export("grad_aware", path)(x)
            self.assertEqual(self._export("grad_aware", path, check=True)(x), x * 2)

    def test_check_defaults_to_the_environment_variable(self):
        path = os.path.join(self.dir, "shifted.py")
        x = torch.randn(8)
        self._export("shifted", path)(x)
        _edit_graph_code(path, "7.25", "7.5")
        _in_a_fresh_process(
            self,
            f"""
            import os
            os.environ["TORCH_EXPORT_PYTHON_CHECK"] = "1"
            run = torch.compiler.export_python(path={path!r})(m.shifted)
            try:
                run(torch.ones(8))
            except torch.compiler.PrecompileError as e:
                if "does not match eager" not in str(e):
                    raise
            else:
                raise AssertionError("TORCH_EXPORT_PYTHON_CHECK=1 did not check")
            """,
        )

    def test_mode_and_options_are_checked_as_torch_compile_checks_them(self):
        export = functools.partial(
            torch.compiler.export_python, os.path.join(self.dir, "f.py")
        )
        with self.assertRaisesRegex(ValueError, "mode or options, not both"):
            export(mode="max-autotune", options={"max_autotune": True})(
                self.module.scaled
            )
        with self.assertRaisesRegex(ValueError, "configure the inductor backend"):
            export(backend="eager", mode="max-autotune")(self.module.scaled)
        with self.assertRaisesRegex(RuntimeError, "Unexpected optimization option"):
            export(options={"not_an_option": True})(self.module.scaled)

    def test_a_main_module_function_is_refused(self):
        def fn(x):
            return x + 1

        fn.__module__ = "__main__"
        with self.assertRaisesRegex(PrecompileError, "importable module"):
            torch.compiler.export_python(path=os.path.join(self.dir, "f.py"))(fn)


@unittest.skipUnless(HAS_CUDA_AND_TRITON, "needs CUDA and Triton")
@skipIfTorchDynamo("precompile captures cannot run under dynamo wrapping")
class TestExportPythonCUDA(_ExportPythonTestCase):
    hw_classification = HardwareClassification.CUDA

    def test_an_edited_kernel_config_is_what_launches(self):
        path = os.path.join(self.dir, "reduce.py")
        x = torch.randn(64, 1024, device="cuda")
        ref = self._export("reduce", path)(x)
        graph_code = _graph_code(path)
        # Compile-time tuning wrote each kernel's launch config into the graph code.
        row = re.search(r"^    '(triton_\w+)': (\{.*\}),$", graph_code, re.MULTILINE)
        self.assertIsNotNone(row, "no KERNEL_CONFIGS row in the graph code")
        name, config = row.group(1), ast.literal_eval(row.group(2))
        warps = 8 if config["num_warps"] != 8 else 4
        edited = {**config, "num_warps": warps}
        _edit_graph_code(path, row.group(0), f"    {name!r}: {edited!r},")
        torch.save((x, ref), os.path.join(self.dir, "io.pt"))
        _in_a_fresh_process(
            self,
            f"""
            from torch._inductor.codecache import PyCodeCache
            x, ref = torch.load({os.path.join(self.dir, "io.pt")!r})
            run = torch.compiler.export_python(path={path!r})(m.reduce)
            torch.testing.assert_close(run(x), ref)
            (module,) = (
                mod
                for mod in PyCodeCache.modules
                if getattr(mod, "KERNEL_CONFIGS", {{}}).get({name!r}) == {edited!r}
            )
            (launcher,) = getattr(module, {name!r}).launchers
            if launcher.config.num_warps != {warps}:
                raise AssertionError(launcher.config)
            """,
        )

    def test_max_autotune_composes(self):
        # mode= is torch.compile's: the capture tunes under max-autotune and the file
        # holds what it chose, served without compiling and without CUDA graphs.
        path = os.path.join(self.dir, "linear_relu.py")
        x, w = (torch.randn(256, 256, device="cuda") for _ in range(2))
        with self.assertWarnsRegex(UserWarning, "CUDA graphs"):
            out = self._export("linear_relu", path, mode="max-autotune")(x, w)
        self.assertEqual(out, (x @ w).relu(), atol=1e-3, rtol=1e-3)
        code = _graph_code(path)
        self.assertIn("'max_autotune': True", code)
        self.assertNotIn("def partition_0", code)
        torch.save((x, w, out), os.path.join(self.dir, "io.pt"))
        _in_a_fresh_process(
            self,
            f"""
            from torch.compiler.precompile import no_compilation
            x, w, out = torch.load({os.path.join(self.dir, "io.pt")!r})
            run = torch.compiler.export_python(path={path!r})(m.linear_relu)
            with no_compilation():
                torch.testing.assert_close(run(x, w), out)
            # A tuned GEMM sums in another order than cuBLAS; the check allows that.
            checked = torch.compiler.export_python(path={path!r}, check=True)
            torch.testing.assert_close(checked(m.linear_relu)(x, w), out)
            """,
        )

    def test_flex_attention_hill_climb_step(self):
        # One step of an agent tuning an exported flex attention kernel: export,
        # change the template's num_warps in the file, rerun with the eager check.
        path = os.path.join(self.dir, "attention.py")
        q, k, v = (torch.randn(1, 2, 256, 64, device="cuda") for _ in range(3))
        expected = self.module.attention(q, k, v)
        out = self._export("attention", path)(q, k, v)
        self.assertEqual(out, expected, atol=2e-3, rtol=2e-3)
        src = _graph_code(path)
        (name,) = re.findall(r"^def (triton_tem_\w+)\(", src, re.MULTILINE)
        self.assertEqual(src.count("num_warps=4,"), 1)
        _edit_graph_code(path, "num_warps=4,", "num_warps=8,")
        torch.save((q, k, v, expected), os.path.join(self.dir, "io.pt"))
        _in_a_fresh_process(
            self,
            f"""
            import inspect
            from torch._inductor.codecache import PyCodeCache
            q, k, v, expected = torch.load({os.path.join(self.dir, "io.pt")!r})
            run = torch.compiler.export_python(path={path!r}, check=True)(m.attention)
            torch.testing.assert_close(run(q, k, v), expected, atol=2e-3, rtol=2e-3)
            (module,) = (
                mod
                for mod in PyCodeCache.modules
                if hasattr(mod, {name!r}) and "num_warps=8," in inspect.getsource(mod)
            )
            (launcher,) = getattr(module, {name!r}).launchers
            if launcher.config.num_warps != 8:
                raise AssertionError(launcher.config)
            """,
        )


if __name__ == "__main__":
    run_tests()

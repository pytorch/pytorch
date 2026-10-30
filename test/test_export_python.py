# Owner(s): ["oncall: pt2"]
import ast
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
from torch.compiler.precompile import no_compilation
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


# export_python serves an artifact by importing the module its function was captured
# from, so each test writes its functions into a fresh importable module.
_MODULE = """
import os

import torch
import torch._dynamo.testing

def scaled(x, *, scale=2.0):
    return (x * scale).relu()

def inner(x):
    y = x.sin()
    print("graph break")
    return y.cos()

def outer(x):
    return inner(x * 2) + 1

live_counter = torch._dynamo.testing.CompileCounter()

@torch.compile(backend=live_counter)
def live(x):
    y = x.sin()
    print("graph break")
    return y.cos()

def calls_live(x):
    return live(x) + 1

def breaks(x):
    y = x.sin()
    print("graph break")
    return y.cos()

def flaky(x, fail):
    y = x.sin()
    print("graph break")
    if fail:
        raise ValueError("first call failed")
    return y.cos()

_ARTIFACTS = os.path.join(os.path.dirname(__file__), "decorated")

@torch.compiler.export_python(path=os.path.join(_ARTIFACTS, "scaled.py"))
def decorated_scaled(x):
    return (x * 2.0).relu()

@torch.compiler.export_python(path=os.path.join(_ARTIFACTS, "outer.py"))
def decorated_outer(x):
    return inner(x * 2) + 1

def shifted(x):
    return x * 7.25 + 1

def reduce(x):
    return (x.sin() * 2).sum(dim=-1)

"""


def _edit_graph_code(path, old, new):
    # Only the compiled graph code, as an agent hill-climbing the file would.
    with open(path) as f:
        src = f.read()
    head, marker, code = src.partition("GRAPH_SOURCES = {}")
    if old not in code:
        raise AssertionError(f"{old!r} not in the graph code of {path}")
    with open(path, "w") as f:
        f.write(head + marker + code.replace(old, new))


def _in_a_fresh_process(test, body):
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

    def test_graph_break_serves_in_the_capturing_process(self):
        path = os.path.join(self.dir, "breaks.py")
        run = self._export("breaks", path)
        x = torch.randn(4)
        self.assertEqual(run(x), x.sin().cos())
        with no_compilation():
            self.assertEqual(run(x), x.sin().cos())
        # The capture's live compile is released, not broken: fn still compiles.
        self.assertEqual(torch.compile(self.module.breaks)(x), x.sin().cos())

    def test_a_graph_break_in_a_called_function_serves_in_the_capturing_process(self):
        # Written as an installed artifact, which refuses frames with live entries.
        path = os.path.join(self.dir, "outer.py")
        run = self._export("outer", path)
        x = torch.randn(4)
        self.assertEqual(run(x), (x * 2).sin().cos() + 1)
        with no_compilation():
            self.assertEqual(run(x), (x * 2).sin().cos() + 1)

    def test_a_live_compile_of_another_function_survives_the_capture(self):
        x = torch.randn(4)
        self.assertEqual(self.module.live(x), x.sin().cos())
        compiles = self.module.live_counter.frame_count
        run = self._export("calls_live", os.path.join(self.dir, "calls_live.py"))
        self.assertEqual(run(x), x.sin().cos() + 1)
        self.assertEqual(self.module.live(x), x.sin().cos())
        # Its frames ran under its own compile, so the capture released none of them.
        self.assertEqual(self.module.live_counter.frame_count, compiles)

    def test_a_capture_beside_a_loaded_artifact_on_a_shared_frame(self):
        # outer and decorated_outer both graph-break in inner, where the installed
        # artifact loaded first keeps its entries while the second is captured.
        x = torch.randn(4)
        expected = (x * 2).sin().cos() + 1
        run = self._export("outer", os.path.join(self.dir, "outer.py"))
        self.assertEqual(run(x), expected)
        self.assertEqual(self.module.decorated_outer(x), expected)
        with no_compilation():
            self.assertEqual(run(x), expected)
            self.assertEqual(self.module.decorated_outer(x), expected)
        # The first artifact served inner during the second capture, which therefore
        # did not compile it: the second artifact is standalone, inner left to Python.
        with open(os.path.join(self.dir, "decorated", "outer.py")) as f:
            self.assertIn('SERVING_MODE = "standalone"', f.read())
        _in_a_fresh_process(
            self,
            """
            from torch.compiler.precompile import no_compilation
            x = torch.randn(4)
            with no_compilation():
                torch.testing.assert_close(m.decorated_outer(x), (x * 2).sin().cos() + 1)
            """,
        )

    def test_a_new_process_serves_the_file_without_compiling(self):
        path = os.path.join(self.dir, "breaks.py")
        self._export("breaks", path)(torch.randn(4))
        _in_a_fresh_process(
            self,
            f"""
            from torch.compiler.precompile import no_compilation
            run = torch.compiler.export_python(path={path!r})(m.breaks)
            x = torch.randn(4)
            with no_compilation():
                torch.testing.assert_close(run(x), x.sin().cos())
            """,
        )

    def test_the_decorator_serves_here_and_in_a_new_process(self):
        # The module binds export_python's wrapper under each decorated name, which
        # loading looks the captured function up through.
        x = torch.randn(4)
        for fn, expected in [
            (self.module.decorated_scaled, (x * 2.0).relu()),
            (self.module.decorated_outer, (x * 2).sin().cos() + 1),
        ]:
            self.assertEqual(fn(x), expected)
            with no_compilation():
                self.assertEqual(fn(x), expected)
        _in_a_fresh_process(
            self,
            """
            from torch.compiler.precompile import no_compilation
            x = torch.randn(4)
            with no_compilation():
                torch.testing.assert_close(m.decorated_scaled(x), (x * 2.0).relu())
                torch.testing.assert_close(m.decorated_outer(x), (x * 2).sin().cos() + 1)
            """,
        )

    def test_a_first_call_that_raises_writes_nothing_and_the_next_call_captures(self):
        path = os.path.join(self.dir, "flaky.py")
        run = self._export("flaky", path)
        x = torch.randn(4)
        with self.assertRaisesRegex(ValueError, "first call failed"):
            run(x, True)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(run(x, False), x.sin().cos())
        with no_compilation():
            self.assertEqual(run(x, False), x.sin().cos())

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

    def test_a_main_module_function_is_refused(self):
        def fn(x):
            return x + 1

        fn.__module__ = "__main__"
        with self.assertRaisesRegex(PrecompileError, "importable module"):
            torch.compiler.export_python(path=os.path.join(self.dir, "f.py"))(fn)


@unittest.skipUnless(HAS_CUDA_AND_TRITON, "needs CUDA and Triton")
class TestExportPythonCUDA(_ExportPythonTestCase):
    hw_classification = HardwareClassification.CUDA

    def test_an_edited_kernel_config_is_what_launches(self):
        path = os.path.join(self.dir, "reduce.py")
        x = torch.randn(64, 1024, device="cuda")
        ref = self._export("reduce", path)(x)
        with open(path) as f:
            graph_code = f.read().partition("GRAPH_SOURCES = {}")[2]
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


if __name__ == "__main__":
    run_tests()

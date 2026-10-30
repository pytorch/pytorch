# Owner(s): ["oncall: pt2"]
import importlib
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid

import torch
from torch.compiler import PrecompileError
from torch.compiler.precompile import no_compilation
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


# export_python serves an artifact by importing the module its function was captured
# from, so each test writes its functions into a fresh importable module.
_MODULE = """
import torch

def scaled(x, *, scale=2.0):
    return (x * scale).relu()

def inner(x):
    y = x.sin()
    print("graph break")
    return y.cos()

def outer(x):
    return inner(x * 2) + 1

@torch.compile(backend="eager")
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
"""


class TestExportPython(TestCase):
    hw_classification = HardwareClassification.GENERIC

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
        run = self._export("calls_live", os.path.join(self.dir, "calls_live.py"))
        self.assertEqual(run(x), x.sin().cos() + 1)
        self.assertEqual(self.module.live(x), x.sin().cos())

    def test_a_new_process_serves_the_file_without_compiling(self):
        path = os.path.join(self.dir, "breaks.py")
        x = torch.randn(4)
        self._export("breaks", path)(x)
        script = textwrap.dedent(
            f"""
            import sys, torch
            sys.path.insert(0, {self.dir!r})
            import {self.module_name} as m
            from torch.compiler.precompile import no_compilation
            run = torch.compiler.export_python(path={path!r})(m.breaks)
            x = torch.load({os.path.join(self.dir, "x.pt")!r})
            with no_compilation():
                torch.testing.assert_close(run(x), x.sin().cos())
            print("served")
            """
        )
        torch.save(x, os.path.join(self.dir, "x.pt"))
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("served", out.stdout)

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
        with self.assertRaisesRegex(PrecompileError, "no captured variant"):
            run(torch.randn(3, 3))

    def test_a_main_module_function_is_refused(self):
        def fn(x):
            return x + 1

        fn.__module__ = "__main__"
        with self.assertRaisesRegex(PrecompileError, "importable module"):
            torch.compiler.export_python(path=os.path.join(self.dir, "f.py"))(fn)


if __name__ == "__main__":
    run_tests()

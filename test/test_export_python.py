# Owner(s): ["oncall: pt2"]
import importlib
import os
import subprocess
import sys
import tempfile
import textwrap
import uuid

import torch
from torch._precompile import _runnable_from_pair
from torch.compiler import PrecompileError
from torch.compiler._export_python import (
    _captured_artifact,
    _CODE_BEGIN,
    _CODE_END,
    _exported_file,
)
from torch.compiler.precompile import (
    capture,
    DynamoTracer,
    MakeFxTracer,
    no_compilation,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)


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

_ARTIFACTS = os.path.join(os.path.dirname(__file__), "decorated")

@torch.compiler.export_python(path=os.path.join(_ARTIFACTS, "scaled.py"))
def decorated_scaled(x):
    return (x * 2.0).relu()

def shifted(x):
    return x * 7.25 + 1

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
class TestExportPythonFile(_ExportPythonTestCase):
    hw_classification = HardwareClassification.GENERIC

    def _artifact(self, name, *args, fullgraph=True, tracer=None, **kwargs):
        # The precompile artifact a capture writes, which the file is made from.
        artifact, cache = (os.path.join(self.dir, name + e) for e in (".a", ".cache"))
        tracer = tracer or DynamoTracer(fullgraph=fullgraph)
        fn = getattr(self.module, name)
        with capture(
            fn, artifact_path=artifact, cache_path=cache, tracer=tracer, **kwargs
        ) as cap:
            cap(*args)
        with open(artifact) as f:
            return f.read(), cache

    def test_the_file_is_the_graph_code_and_one_captured_line(self):
        path = os.path.join(self.dir, "scaled.py")
        with open(path, "w") as f:
            f.write(_exported_file(self._artifact("scaled", torch.randn(8))[0]))
        with open(path) as f:
            lines = f.read().splitlines()
        self.assertTrue(
            lines[0].startswith("# Exported by torch.compiler.export_python")
        )
        self.assertIn("def call(", _graph_code(path))
        self.assertTrue(lines[-1].startswith("_CAPTURED = '"))
        # Nothing between the graph code and that line but a comment, and none of the
        # precompile driver anywhere.
        tail = lines[lines.index(_CODE_END) + 1 : -1]
        self.assertTrue(all(not line or line.startswith("#") for line in tail), tail)
        self.assertNotIn("_build_multigraph_forward", "\n".join(lines))

    def test_the_file_rebuilds_the_artifact_it_was_made_from(self):
        artifact, cache = self._artifact("scaled", torch.randn(8))
        rebuilt = _captured_artifact(_exported_file(artifact), "scaled.py")
        self.assertEqual(rebuilt, artifact)
        # So the capture's cache, whose code_hash covers the artifact, serves it.
        with open(cache, "rb") as f:
            run = _runnable_from_pair(rebuilt, f.read())
        x = torch.randn(8)
        with no_compilation():
            self.assertEqual(run(x), (x * 2.0).relu())

    def test_the_artifact_takes_the_graph_code_the_file_holds(self):
        artifact, _ = self._artifact("scaled", torch.randn(8))
        text = _exported_file(artifact).replace(_CODE_END, "# an edit\n" + _CODE_END)
        rebuilt = _captured_artifact(text, "scaled.py")
        self.assertNotEqual(rebuilt, artifact)
        self.assertIn("# an edit", rebuilt)

    def test_an_eager_capture_has_no_graph_code(self):
        artifact, _ = self._artifact("scaled", torch.randn(8), backend="eager")
        text = _exported_file(artifact)
        self.assertNotIn(_CODE_BEGIN, text)
        self.assertEqual(_captured_artifact(text, "scaled.py"), artifact)

    def test_a_capture_that_is_not_one_graph_is_refused(self):
        x = torch.randn(4)
        for backend in ("inductor", "eager"):
            artifact, _ = self._artifact("breaks", x, fullgraph=False, backend=backend)
            with self.assertRaisesRegex(PrecompileError, "is not one graph"):
                _exported_file(artifact)

    def test_a_make_fx_capture_is_refused(self):
        artifact, _ = self._artifact("scaled", torch.randn(8), tracer=MakeFxTracer())
        with self.assertRaisesRegex(PrecompileError, "is not one graph"):
            _exported_file(artifact)

    def test_a_damaged_file_is_refused(self):
        text = _exported_file(self._artifact("scaled", torch.randn(8))[0])
        with self.assertRaisesRegex(PrecompileError, "no _CAPTURED line"):
            _captured_artifact(text.rpartition("_CAPTURED")[0], "f.py")
        with self.assertRaisesRegex(PrecompileError, "lost the lines"):
            _captured_artifact(text.replace(_CODE_BEGIN, "#"), "f.py")
        corrupt = text.replace("_CAPTURED = 'eN", "_CAPTURED = 'xN")
        with self.assertRaisesRegex(
            PrecompileError, "_CAPTURED line of f.py is corrupt"
        ):
            _captured_artifact(corrupt, "f.py")

    def test_running_the_file_benchmarks_it(self):
        path = os.path.join(self.dir, "scaled.py")
        text = _exported_file(self._artifact("scaled", torch.randn(8))[0])
        self.assertRegex(text, r"Running this file\s+(#\s+)?benchmarks it")
        with open(path, "w") as f:
            f.write(text)
        out = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=900
        )
        self.assertEqual(out.returncode, 0, out.stderr)


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

    def test_only_a_file_this_process_did_not_write_gets_the_exec_warning(self):
        path = os.path.join(self.dir, "scaled.py")
        with self.assertNoLogs("torch._precompile", "WARNING"):
            self._export("scaled", path)(torch.randn(8))
        with self.assertLogs("torch._precompile", "WARNING") as logs:
            self._export("scaled", path)(torch.randn(8))
        self.assertIn("about to EXEC", logs.output[0])

    def test_a_graph_break_is_refused(self):
        path = os.path.join(self.dir, "breaks.py")
        with self.assertRaisesRegex(PrecompileError, "does not trace as one graph"):
            self._export("breaks", path)(torch.randn(4))
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".cache"))

    def test_a_call_that_needs_a_backward_graph_is_refused(self):
        path = os.path.join(self.dir, "scaled.py")
        run = self._export("scaled", path)
        x = torch.randn(8, requires_grad=True)
        with self.assertRaisesRegex(PrecompileError, "backward graph too"):
            run(x)
        self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(path + ".cache"))
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
        self.assertFalse(os.path.exists(path + ".cache"))
        self.assertEqual(run(x, y), x @ y)
        with no_compilation():
            self.assertEqual(run(x, y), x @ y)

    def test_the_eager_backend_writes_no_graph_code(self):
        path = os.path.join(self.dir, "scaled.py")
        x = torch.randn(8)
        run = self._export("scaled", path, backend="eager")
        self.assertEqual(run(x), x.mul(2).relu())
        with open(path) as f:
            self.assertNotIn(_CODE_BEGIN, f.read())
        with no_compilation():
            self.assertEqual(self._export("scaled", path)(x), x.mul(2).relu())

    def test_dynamic_serves_another_shape(self):
        path = os.path.join(self.dir, "scaled.py")
        self._export("scaled", path, dynamic=True)(torch.randn(8))
        y = torch.randn(16)
        with no_compilation():
            self.assertEqual(self._export("scaled", path)(y), (y * 2.0).relu())

    def test_a_file_captured_from_another_function_is_refused(self):
        path = os.path.join(self.dir, "shared.py")
        self._export("scaled", path)(torch.randn(8))
        x, y = torch.randn(2, 3), torch.randn(3, 2)
        with self.assertRaisesRegex(PrecompileError, r"scaled, not .*\.matmul"):
            self._export("matmul", path)(x, y)

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


if __name__ == "__main__":
    run_tests()

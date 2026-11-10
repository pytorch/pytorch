# Owner(s): ["oncall: pt2"]
import importlib
import os
import sys
import tempfile
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
from torch.compiler.precompile import capture, DynamoTracer, no_compilation
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


# Serving an artifact imports the module its function was captured from, so each
# test writes its functions into a fresh importable module.
_MODULE = """
import torch

def scaled(x, *, scale=2.0):
    return (x * scale).relu()

def breaks(x):
    y = x.sin()
    print("graph break")
    return y.cos()
"""


def _graph_code(path):
    with open(path) as f:
        return f.read().partition(_CODE_BEGIN)[2].rpartition(_CODE_END)[0]


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


@skipIfTorchDynamo("precompile captures cannot run under dynamo wrapping")
class TestExportPythonFile(_ExportPythonTestCase):
    hw_classification = HardwareClassification.GENERIC

    def _artifact(self, name, *args, fullgraph=True, **kwargs):
        # The precompile artifact a capture writes, which the file is made from.
        artifact, cache = (os.path.join(self.dir, name + e) for e in (".a", ".cache"))
        tracer = DynamoTracer(fullgraph=fullgraph)
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
        artifact, _ = self._artifact("breaks", torch.randn(4), fullgraph=False)
        with self.assertRaisesRegex(PrecompileError, "is not one graph"):
            _exported_file(artifact)

    def test_a_file_that_lost_its_captured_line_or_markers_is_refused(self):
        text = _exported_file(self._artifact("scaled", torch.randn(8))[0])
        with self.assertRaisesRegex(PrecompileError, "no _CAPTURED line"):
            _captured_artifact(text.rpartition("_CAPTURED")[0], "f.py")
        with self.assertRaisesRegex(PrecompileError, "lost the lines"):
            _captured_artifact(text.replace(_CODE_BEGIN, "#"), "f.py")


if __name__ == "__main__":
    run_tests()

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@instantiate_parametrized_tests
class TestRamjetMetadata(TestCase):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory(prefix="ramjet-metadata-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.root.joinpath("test").mkdir()
        for path in ("pytest.ini", "test/conftest.py", "test/pytest_shard_custom.py"):
            shutil.copyfile(REPO_ROOT / path, self.root / path)
        self.source = self.root / "test/test_example.py"
        self.source.write_text(
            textwrap.dedent("""\
            import os
            from torch.testing._internal import tci
            from torch.testing._internal.common_utils import run_tests, TestCase

            os.write(1, b"native module output\\n")

            class TestExample(TestCase):
                def test_selected(self):
                    pass

                def test_other(self):
                    raise AssertionError("unselected test ran")

                @tci.test(exclude=tci.platform.linux_x86_64)
                def test_cuda(self):
                    raise AssertionError("CUDA test ran on CPU")

            if __name__ == "__main__":
                run_tests()
            """)
        )

    def invoke(self, *args, script=False, selection=None, output=True, launcher=False):
        env = dict(os.environ)
        env.pop("PYTORCH_TEST_SELECTION_FILE", None)
        env.pop("PYTORCH_TEST_LIST_MARKERS_FILE", None)
        env.pop("PYTORCH_TEST_COLLECTION_STATUS_FILE", None)
        if selection is not None:
            selection_file = self.root / "selection.json"
            selection_file.write_text(json.dumps(selection))
            env["PYTORCH_TEST_SELECTION_FILE"] = str(selection_file)
        if output:
            env["PYTORCH_TEST_LIST_MARKERS_FILE"] = str(self.root / "metadata.json")
        command = (
            ["test/test_example.py", "--use-pytest"]
            if script
            else ["-m", "pytest", "test/test_example.py"]
        )
        if launcher:
            command = [
                str(REPO_ROOT / "tools/ramjet_run.py"),
                "test/test_example.py",
                "--use-pytest",
            ]
            env["PYTORCH_TEST_COLLECTION_STATUS_FILE"] = str(self.root / "status.json")
        return subprocess.run(
            [sys.executable, *command, *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )

    @parametrize("script", [False, True])
    def test_metadata_file_preserves_native_stdout(self, script):
        result = self.invoke("--list-markers", script=script)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native module output", result.stdout)
        metadata = json.loads(self.root.joinpath("metadata.json").read_text())
        self.assertEqual(len(metadata), 3)
        self.assertIn(
            "tci.platform:linux_x86_64",
            metadata["test/test_example.py::TestExample::test_selected"],
        )

    def test_stdout_fallback(self):
        self.source.write_text(
            self.source.read_text().replace(
                'os.write(1, b"native module output\\n")', ""
            )
        )
        result = self.invoke("--list-markers", output=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(json.loads(result.stdout)), 3)

    def test_cpu_wrapper_capabilities_do_not_change_collected_nodes(self):
        self.source.write_text(
            textwrap.dedent(f"""\
            import sys
            from unittest.mock import patch

            import torch
            from torch.testing._internal.common_utils import run_tests

            sys.path.insert(0, {str(REPO_ROOT / "test/inductor")!r})
            with (
                patch.object(torch.ops.mkldnn, "_is_mkldnn_bf16_supported", return_value=False),
                patch.object(torch.ops.mkldnn, "_is_mkldnn_fp16_supported", return_value=False),
            ):
                from test_cpu_cpp_wrapper import (
                    DynamicShapesCppWrapperCpuTests,
                    TestCppWrapper,
                )

            if __name__ == "__main__":
                run_tests()
            """)
        )
        nodes = [
            f"test/test_example.py::{test_class}::test_linear_{kind}{suffix}_cpp_wrapper"
            for test_class, suffix in (
                ("TestCppWrapper", ""),
                ("DynamicShapesCppWrapperCpuTests", "_dynamic_shapes"),
            )
            for kind in ("binary", "packed")
        ]
        result = self.invoke("--list-markers", script=True, selection=nodes)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            set(json.loads(self.root.joinpath("metadata.json").read_text())),
            set(nodes),
        )
        result = self.invoke("-q", script=True, selection=nodes, output=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("4 skipped", result.stdout)

    @parametrize("script", [False, True])
    def test_selection_file(self, script):
        node = "test/test_example.py::TestExample::test_selected"
        result = self.invoke("--list-markers", script=script, selection=[node])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            list(json.loads(self.root.joinpath("metadata.json").read_text())), [node]
        )
        result = self.invoke("-q", script=script, selection=[node])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 passed", result.stdout)

    @parametrize("selection", [[], {}, [1], [""]])
    def test_invalid_selection_is_an_error(self, selection):
        result = self.invoke("--list-markers", selection=selection)
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("nonempty JSON list of node IDs", result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())

    @parametrize("script", [False, True])
    def test_missing_selection_is_an_error(self, script):
        result = self.invoke(
            "--list-markers",
            script=script,
            selection=["test/test_example.py::TestExample::test_missing"],
        )
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("test_missing", result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())

    def test_collection_reports_preserve_tracebacks(self):
        self.source.write_text(
            'raise RuntimeError("metadata collection import failed")\n'
        )
        result = self.invoke("--list-markers")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("RuntimeError: metadata collection import failed", result.stderr)
        self.assertIn("test_example.py", result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())

    def test_selection_preserves_colons_in_test_names(self):
        self.source.write_text(
            textwrap.dedent("""\
            from torch.testing._internal.common_utils import run_tests, TestCase

            class TestExample(TestCase):
                pass

            setattr(TestExample, "test_expected_at::kCPU", lambda self: None)

            if __name__ == "__main__":
                run_tests()
            """)
        )
        node = "test/test_example.py::TestExample::test_expected_at::kCPU"
        result = self.invoke("--list-markers", script=True, selection=[node])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            list(json.loads(self.root.joinpath("metadata.json").read_text())), [node]
        )
        result = self.invoke("-q", script=True, selection=[node])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 passed", result.stdout)

    def test_class_selection(self):
        result = self.invoke(
            "--list-markers", selection=["test/test_example.py::TestExample"]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            len(json.loads(self.root.joinpath("metadata.json").read_text())), 3
        )

    def test_empty_file_selection(self):
        self.source.write_text("pass\n")
        result = self.invoke("--list-markers", selection=["test/test_example.py"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(self.root.joinpath("metadata.json").read_text()), {}
        )

    def test_selection_exceeding_argument_limit(self):
        name = "test_" + "x" * 150_000
        self.source.write_text(
            "from torch.testing._internal.common_utils import TestCase\n"
            "class TestExample(TestCase):\n"
            "    pass\n"
            f"setattr(TestExample, {name!r}, lambda self: None)\n"
        )
        node = f"test/test_example.py::TestExample::{name}"
        result = self.invoke("--list-markers", selection=[node])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            list(json.loads(self.root.joinpath("metadata.json").read_text())), [node]
        )

    def test_selection_respects_tci_platform(self):
        prefix = "test/test_example.py::TestExample::"
        with mock.patch.dict(os.environ, PYTORCH_TEST_TCI_PLATFORM="linux_x86_64"):
            result = self.invoke(
                "-q",
                script=True,
                selection=[prefix + "test_selected", prefix + "test_cuda"],
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 passed, 2 deselected", result.stdout)

    def test_parameterized_function_selection(self):
        self.source.write_text(
            "import pytest\n"
            "@pytest.mark.parametrize('value', [1, 2])\n"
            "def test_parameterized(value):\n"
            "    pass\n"
        )
        node = "test/test_example.py::test_parameterized"
        result = self.invoke("--list-markers", selection=[node])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            set(json.loads(self.root.joinpath("metadata.json").read_text())),
            {f"{node}[1]", f"{node}[2]"},
        )

    def test_selection_does_not_leak_to_nested_pytest(self):
        self.root.joinpath("test/test_nested.py").write_text(
            "from torch.testing._internal.common_utils import TestCase\n"
            "class TestNested(TestCase):\n"
            "    def test_nested(self):\n"
            "        pass\n"
        )
        self.source.write_text(
            textwrap.dedent("""\
            import os
            import subprocess
            import sys
            from torch.testing._internal.common_utils import run_tests, TestCase

            class TestExample(TestCase):
                def test_selected(self):
                    self.assertNotIn("PYTORCH_TEST_SELECTION_FILE", os.environ)
                    result = subprocess.run(
                        [sys.executable, "-m", "pytest", "test/test_nested.py", "-q"],
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("1 passed", result.stdout)

                def test_other(self):
                    self.fail("unselected test ran")

            if __name__ == "__main__":
                run_tests()
            """)
        )
        result = self.invoke(
            "-q",
            script=True,
            selection=["test/test_example.py::TestExample::test_selected"],
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 passed", result.stdout)

    def test_pytest_lint_does_not_construct_tests(self):
        self.source.write_text(
            textwrap.dedent("""\
            from torch.testing._internal.common_utils import run_tests, TestCase

            class TestExample(TestCase):
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("test constructor must not run during collection")

                def test_example(self):
                    pass

            if __name__ == "__main__":
                run_tests()
            """)
        )
        result = self.invoke("--list-markers", script=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            len(json.loads(self.root.joinpath("metadata.json").read_text())), 1
        )

    def test_pytest_lint_still_rejects_wrong_test_base(self):
        self.source.write_text(
            textwrap.dedent("""\
            import unittest
            from torch.testing._internal.common_utils import run_tests

            class TestExample(unittest.TestCase):
                def test_example(self):
                    pass

            if __name__ == "__main__":
                run_tests()
            """)
        )
        result = self.invoke("--list-markers", script=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("should extend from", result.stdout)
        self.assertFalse(self.root.joinpath("metadata.json").exists())

    @parametrize(
        "code",
        [
            "raise SystemExit(0)",
            "raise SystemExit",
            "raise unittest.SkipTest('requires accelerator')",
        ],
    )
    def test_launcher_records_explicit_module_skip(self, code):
        self.source.write_text(f"import unittest\n{code}\n")
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status = json.loads(self.root.joinpath("status.json").read_text())
        self.assertTrue(status["skipped"])
        self.assertEqual(status["filename"], str(self.source))
        self.assertEqual(status["line"], 2)
        self.assertFalse(self.root.joinpath("metadata.json").exists())

    @parametrize(
        "code,returncode",
        [
            ("pass", 0),
            ("raise SystemExit(1)", 1),
            ("raise RuntimeError('broken module')", 1),
        ],
    )
    def test_launcher_does_not_invent_skip_status(self, code, returncode):
        self.source.write_text(code + "\n")
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, returncode, result.stdout + result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())
        self.assertFalse(self.root.joinpath("status.json").exists())

    def test_launcher_preserves_metadata(self):
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            len(json.loads(self.root.joinpath("metadata.json").read_text())), 3
        )
        self.assertFalse(self.root.joinpath("status.json").exists())

    def test_launcher_does_not_treat_runner_exit_as_skip(self):
        self.source.write_text(
            "from unittest import mock\n"
            "from torch.testing._internal.common_utils import run_tests\n"
            "with mock.patch('pytest.main', return_value=0):\n"
            "    run_tests()\n"
        )
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())
        self.assertFalse(self.root.joinpath("status.json").exists())

    @parametrize(
        "need,reason",
        [
            ("cuda", "requires CUDA"),
            ("ramjet_missing_test_dependency", "cannot be imported"),
        ],
    )
    def test_dynamo_dependency_skip_records_reason(self, need, reason):
        self.source.write_text(
            "from unittest import mock\n"
            "from torch._dynamo.test_case import run_tests\n"
            "with mock.patch('torch.cuda.is_available', return_value=False):\n"
            f"    run_tests(needs={need!r})\n"
        )
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        status = json.loads(self.root.joinpath("status.json").read_text())
        self.assertIn(reason, status["skipped"])
        self.assertTrue(status["filename"].endswith("torch/_dynamo/test_case.py"))
        self.assertFalse(self.root.joinpath("metadata.json").exists())

    def test_hardware_main_guard_collects_and_runs_selected_cpu_tests(self):
        self.source.write_text(
            self.source.read_text().replace(
                'if __name__ == "__main__":\n    run_tests()',
                'HAS_GPU = False\nif __name__ == "__main__":\n    if HAS_GPU:\n        run_tests()',
            )
        )
        node = "test/test_example.py::TestExample::test_selected"
        result = self.invoke("--list-markers", launcher=True, selection=[node])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            list(json.loads(self.root.joinpath("metadata.json").read_text())), [node]
        )
        self.root.joinpath("metadata.json").unlink()
        with mock.patch.dict(os.environ, PYTORCH_TEST_TCI_PLATFORM="linux_x86_64"):
            result = self.invoke("-q", launcher=True, selection=[node], output=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 passed", result.stdout)

    @parametrize(
        "guard",
        ["HAS_GPU and sys.version_info < (0, 0)", "torch.backends.mps.is_available()"],
    )
    def test_nonhardware_and_other_platform_guards_stay_skipped(self, guard):
        self.source.write_text(
            "import sys\nimport torch\n"
            "from torch.testing._internal.common_utils import run_tests\n"
            "HAS_GPU = False\n"
            "if __name__ == '__main__':\n"
            f"    if {guard}:\n"
            "        run_tests()\n"
        )
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())
        status = json.loads(self.root.joinpath("status.json").read_text())
        self.assertIn("Main runner requires", status["skipped"])

    def test_hardware_main_setup_is_not_bypassed(self):
        self.source.write_text(
            "from torch.testing._internal.common_utils import run_tests\n"
            "HAS_GPU = False\n"
            "if __name__ == '__main__':\n"
            "    if HAS_GPU:\n"
            "        raise RuntimeError('GPU setup cannot run on CPU')\n"
            "        run_tests()\n"
        )
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())
        status = json.loads(self.root.joinpath("status.json").read_text())
        self.assertIn("HAS_GPU", status["skipped"])

    def test_unknown_guard_function_is_not_reexecuted(self):
        self.source.write_text(
            "from torch.testing._internal.common_utils import run_tests\n"
            "def condition():\n"
            "    print('condition evaluated')\n"
            "    return False\n"
            "if __name__ == '__main__':\n"
            "    if condition():\n"
            "        run_tests()\n"
        )
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("condition evaluated"), 1)
        self.assertFalse(self.root.joinpath("metadata.json").exists())
        self.assertFalse(self.root.joinpath("status.json").exists())

    @parametrize("setup", ["undefined_gpu_setup()", "gpu.prepare()"])
    def test_guarded_setup_callee_may_be_unavailable(self, setup):
        self.source.write_text(
            "from types import SimpleNamespace\n"
            "from torch.testing._internal.common_utils import run_tests\n"
            "HAS_GPU = False\n"
            "gpu = SimpleNamespace()\n"
            "if __name__ == '__main__':\n"
            "    if HAS_GPU:\n"
            f"        {setup}\n"
            "        run_tests()\n"
        )
        result = self.invoke("--list-markers", launcher=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.root.joinpath("metadata.json").exists())
        status = json.loads(self.root.joinpath("status.json").read_text())
        self.assertIn("HAS_GPU", status["skipped"])


if __name__ == "__main__":
    run_tests()

# Owner(s): ["module: tests"]

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from td_tracer import (
    _atomic_write_json,
    _run_dir,
    finalize_td_tracer,
    merge_td_tracer_fragments,
    TD_TRACER_CURRENT_TEST_ENV,
    TD_TRACER_DEFER_MERGE_ENV,
    TD_TRACER_RUN_ID_ENV,
    TDTracer,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.testing.merge_td_tracer import merge_td_tracer_outputs


sys.path.remove(str(REPO_ROOT))


@unittest.skipUnless(hasattr(sys, "monitoring"), "requires Python 3.12+")
class TestTDTracerIntegration(TestCase):
    def _write_project(
        self,
        project_dir: Path,
        files: dict[str, str],
        conftest_extra: str = "",
    ) -> None:
        conftest = f"""
import sys
from types import MethodType

from _pytest.python import Module
from td_tracer import TD_TRACER_OPTION_NAME, td_tracer_arguments, TDTracer

def pytest_addoption(parser):
    td_tracer_arguments(parser)
    parser.addoption("--use-main-module", action="store_true")

def pytest_configure(config):
    if config.getoption(TD_TRACER_OPTION_NAME):
        config.pluginmanager.register(TDTracer(config), "pytesttraceplugin")

def pytest_pycollect_makemodule(module_path, parent):
    if parent.config.getoption("--use-main-module"):
        module = Module.from_parent(parent, path=module_path)
        module._getobj = MethodType(lambda self: sys.modules["__main__"], module)
        return module

{conftest_extra}
"""
        (project_dir / "conftest.py").write_text(
            textwrap.dedent(conftest), encoding="utf-8"
        )
        for filename, contents in files.items():
            (project_dir / filename).write_text(
                textwrap.dedent(contents), encoding="utf-8"
            )

    def _subprocess_env(
        self,
        run_id: str | None = None,
        *,
        defer_merge: bool = False,
        parent_test: str | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.pop(TD_TRACER_CURRENT_TEST_ENV, None)
        env.pop(TD_TRACER_DEFER_MERGE_ENV, None)
        env.pop("PYTEST_XDIST_TESTRUNUID", None)
        if run_id is None:
            env.pop(TD_TRACER_RUN_ID_ENV, None)
        else:
            env[TD_TRACER_RUN_ID_ENV] = run_id
        if defer_merge:
            env[TD_TRACER_DEFER_MERGE_ENV] = "1"
        if parent_test is not None:
            env[TD_TRACER_CURRENT_TEST_ENV] = parent_test
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(REPO_ROOT / "test"), env.get("PYTHONPATH")])
        )
        return env

    def _run_project(
        self,
        files: dict[str, str],
        *,
        conftest_extra: str = "",
        extra_args: tuple[str, ...] = (),
        expected_returncode: int = 0,
        parent_test: str | None = None,
    ) -> dict:
        with TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            self._write_project(project_dir, files, conftest_extra)

            output_path = project_dir / "td.json"
            env = self._subprocess_env(parent_test=parent_test)
            test_files = sorted(
                filename for filename in files if filename.startswith("test_")
            )
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *test_files,
                "--td-tracer",
                str(output_path),
                *extra_args,
            ]
            result = subprocess.run(
                command,
                cwd=project_dir,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                expected_returncode,
                msg=result.stdout + result.stderr,
            )
            self.assertTrue(output_path.exists(), msg=result.stdout + result.stderr)
            with output_path.open(encoding="utf-8") as input_file:
                return json.load(input_file)

    def test_collection_and_fixture_coverage(self):
        artifact = self._run_project(
            {
                "test_a.py": """
from torch.utils.benchmark.utils.common import trim_sigfig

trim_sigfig(1.234, 2)

def test_uses_fixture(shared_fixture):
    pass

def test_without_fixture():
    pass
""",
                "test_b.py": """
def test_uses_fixture(shared_fixture):
    pass
""",
                "test_zero.py": """
def test_zero():
    pass
""",
            },
            conftest_extra="""
import pytest

@pytest.fixture(scope="session")
def shared_fixture():
    from torch._dynamo.current_scope_id import current_scope_id
    current_scope_id()
    yield
    from torch._inductor.await_utils import get_loop
    with get_loop(always_create_new_loop=True):
        pass
""",
        )

        coverage = artifact["coverage_by_test"]
        uses_a = coverage["test_a.py::test_uses_fixture"]
        no_fixture = coverage["test_a.py::test_without_fixture"]
        uses_b = coverage["test_b.py::test_uses_fixture"]
        zero = coverage["test_zero.py::test_zero"]

        collection_path = "torch/utils/benchmark/utils/common.py"
        setup_path = "torch/_dynamo/current_scope_id.py"
        teardown_path = "torch/_inductor/await_utils.py"
        self.assertIn(collection_path, uses_a)
        self.assertIn(collection_path, no_fixture)
        self.assertNotIn(collection_path, uses_b)
        self.assertNotIn(collection_path, zero)
        for fixture_path in (setup_path, teardown_path):
            self.assertIn(fixture_path, uses_a)
            self.assertIn(fixture_path, uses_b)
            self.assertNotIn(fixture_path, no_fixture)
            self.assertNotIn(fixture_path, zero)
        self.assertTrue(artifact["complete"])
        self.assertTrue(artifact["successful"])
        self.assertTrue(artifact["usable"])

    def test_zero_edge_test_is_preserved(self):
        artifact = self._run_project(
            {
                "test_zero.py": """
def test_zero():
    pass
"""
            }
        )
        self.assertEqual(artifact["coverage_by_test"], {"test_zero.py::test_zero": []})

    def test_preloaded_modules_are_not_dependencies(self):
        artifact = self._run_project(
            {
                "test_zero.py": """
def test_zero():
    pass
"""
            },
            conftest_extra="""
import sys
from pathlib import Path
from types import ModuleType

import td_tracer

repo_root = Path(td_tracer.__file__).parents[1]
preloaded = ModuleType("preloaded_torch_probe")
preloaded.__file__ = str(repo_root / "torch" / "_utils.py")
sys.modules[preloaded.__name__] = preloaded

namespace = {}
exec(
    compile(
        "def startup_probe(): pass",
        str(repo_root / "torch" / "_environment.py"),
        "exec",
    ),
    namespace,
)

def pytest_sessionstart(session):
    namespace["startup_probe"]()
""",
        )

        self.assertEqual(artifact["coverage_by_test"], {"test_zero.py::test_zero": []})
        self.assertNotIn("global_coverage", artifact)

    def test_no_tests_collected_is_successful(self):
        artifact = self._run_project(
            {
                "test_zero.py": """
def test_zero():
    pass
"""
            },
            extra_args=("-k", "missing"),
            expected_returncode=int(pytest.ExitCode.NO_TESTS_COLLECTED),
        )

        self.assertTrue(artifact["complete"])
        self.assertTrue(artifact["successful"])
        self.assertTrue(artifact["usable"])

    def test_shared_collection_coverage_is_scoped_to_session_tests(self):
        artifact = self._run_project(
            {
                "test_one.py": """
def test_one():
    pass
""",
                "test_two.py": """
def test_two():
    pass
""",
            },
            conftest_extra="""
from pathlib import Path

import td_tracer

namespace = {}
exec(
    compile(
        "def collection_probe(): pass",
        str(Path(td_tracer.__file__).parents[1] / "torch" / "_environment.py"),
        "exec",
    ),
    namespace,
)

def pytest_collection_modifyitems(items):
    namespace["collection_probe"]()
""",
        )

        coverage = artifact["coverage_by_test"]
        self.assertEqual(coverage["test_one.py::test_one"], ["torch/_environment.py"])
        self.assertEqual(coverage["test_two.py::test_two"], ["torch/_environment.py"])

    def test_parent_test_collects_child_dependencies(self):
        artifact = self._run_project(
            {
                "test_one.py": """
def test_one():
    from torch._dynamo.current_scope_id import current_scope_id
    current_scope_id()
"""
            },
            parent_test="logical-suite",
        )

        dependency = "torch/_dynamo/current_scope_id.py"
        self.assertIn(dependency, artifact["coverage_by_test"]["logical-suite"])
        self.assertIn(dependency, artifact["coverage_by_test"]["test_one.py::test_one"])

    @parametrize("joined_option", [False, True])
    def test_subprocess_discovery_does_not_create_tracer_fragment(self, joined_option):
        with TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            self._write_project(
                project_dir,
                {
                    "test_subprocess.py": """
from torch.testing._internal.common_utils import run_tests, TestCase

class TestSubprocess(TestCase):
    def test_one(self):
        from torch._dynamo.current_scope_id import current_scope_id
        current_scope_id()

    def test_two(self):
        from torch._inductor.await_utils import get_loop
        with get_loop(always_create_new_loop=True):
            pass

if __name__ == "__main__":
    run_tests()
""",
                },
            )
            output_path = project_dir / "td.json"
            tracer_args = (
                [f"--td-tracer={output_path}"]
                if joined_option
                else ["--td-tracer", str(output_path)]
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "test_subprocess.py",
                    "--subprocess",
                    "--use-pytest",
                    *tracer_args,
                ],
                cwd=project_dir,
                env=self._subprocess_env("subprocess-run"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertTrue(output_path.exists(), msg=result.stdout + result.stderr)
            with output_path.open(encoding="utf-8") as input_file:
                artifact = json.load(input_file)

        self.assertTrue(artifact["usable"])
        self.assertEqual(len(artifact["participants"]), 2)
        self.assertEqual(
            set(artifact["coverage_by_test"]),
            {
                "test_subprocess.py::TestSubprocess::test_one",
                "test_subprocess.py::TestSubprocess::test_two",
            },
        )

    def test_xdoctest_dependencies_include_logical_parent(self):
        with TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            self._write_project(
                project_dir,
                {
                    "example.py": """
def example():
    \"\"\"
    Example:
        >>> from torch._dynamo.current_scope_id import current_scope_id
        >>> current_scope_id()
    \"\"\"
"""
                },
            )
            output_path = project_dir / "td.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "xdoctest.plugin",
                    "example.py",
                    "--xdoctest-modules",
                    "--td-tracer",
                    str(output_path),
                ],
                cwd=project_dir,
                env=self._subprocess_env("xdoctest", parent_test="doctests"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            with output_path.open(encoding="utf-8") as input_file:
                artifact = json.load(input_file)

        dependency = "torch/_dynamo/current_scope_id.py"
        self.assertIn(dependency, artifact["coverage_by_test"]["doctests"])
        xdoctest_items = [
            test
            for test in artifact["coverage_by_test"]
            if test.startswith("example.py::")
        ]
        self.assertEqual(len(xdoctest_items), 1)
        self.assertIn(dependency, artifact["coverage_by_test"][xdoctest_items[0]])

    def test_incomplete_run_is_not_usable(self):
        artifact = self._run_project(
            {
                "test_failure.py": """
def test_failure():
    assert False

def test_not_run():
    pass
"""
            },
            extra_args=("-x",),
            expected_returncode=1,
        )

        self.assertNotIn("scheduled_tests", artifact)
        self.assertNotIn("completed_tests", artifact)
        self.assertFalse(artifact["complete"])
        self.assertFalse(artifact["successful"])
        self.assertFalse(artifact["usable"])

    def test_independent_pytest_processes_merge(self):
        with TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            files = {
                "test_one.py": """
def test_one():
    pass
""",
                "test_two.py": """
def test_two():
    pass
""",
            }
            self._write_project(project_dir, files)
            output_path = project_dir / "td.json"
            env = self._subprocess_env("shared-run")
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        filename,
                        "--td-tracer",
                        str(output_path),
                    ],
                    cwd=project_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for filename in sorted(files)
            ]
            for process in processes:
                output, _ = process.communicate()
                self.assertEqual(process.returncode, 0, msg=output)

            with output_path.open(encoding="utf-8") as input_file:
                artifact = json.load(input_file)

        self.assertEqual(
            set(artifact["coverage_by_test"]),
            {"test_one.py::test_one", "test_two.py::test_two"},
        )
        self.assertEqual(len(artifact["participants"]), 2)
        self.assertTrue(artifact["usable"])

    def test_deferred_merge_only_writes_fragments(self):
        with TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            self._write_project(
                project_dir,
                {
                    "test_one.py": """
def test_one():
    pass
"""
                },
            )
            output_path = project_dir / "td.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "test_one.py",
                    "--td-tracer",
                    str(output_path),
                ],
                cwd=project_dir,
                env=self._subprocess_env("deferred", defer_merge=True),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertFalse(output_path.exists())

            artifact = merge_td_tracer_fragments(output_path, "deferred")

        self.assertTrue(artifact["usable"])
        self.assertEqual(set(artifact["coverage_by_test"]), {"test_one.py::test_one"})


class TestTDTracerArtifacts(TestCase):
    def _fragment(
        self,
        run_id: str,
        participant_id: str,
        coverage: dict[str, list[str]],
    ) -> dict:
        return {
            "schema_version": 4,
            "run_id": run_id,
            "session_id": participant_id,
            "participant_id": participant_id,
            "worker_id": "test",
            "pid": 1,
            "state": "finished",
            "exit_status": 0,
            "complete": True,
            "environment": {"python": "test", "revision": "revision"},
            "coverage_by_test": coverage,
        }

    def _shard_output(
        self,
        run_id: str,
        coverage: dict[str, list[str]],
        *,
        complete: bool = True,
        successful: bool = True,
    ) -> dict:
        return {
            "schema_version": 4,
            "run_id": run_id,
            "complete": complete,
            "successful": successful,
            "usable": complete and successful,
            "running_participants": [],
            "participants": [
                {
                    "complete": complete,
                    "exit_status": 0 if successful else 1,
                    "participant_id": run_id,
                    "pid": 1,
                    "session_id": run_id,
                    "worker_id": "test",
                }
            ],
            "environments": [{"python": "3.12", "revision": "revision"}],
            "coverage_by_test": coverage,
        }

    def test_merge_unions_fragments_and_tracks_running_participants(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "td.json"
            run_id = "run"
            fragment_dir = _run_dir(output_path, run_id)
            _atomic_write_json(
                fragment_dir / "one.json",
                self._fragment(run_id, "one", {"test_a": ["torch/a.py"]}),
            )
            _atomic_write_json(
                fragment_dir / "two.json",
                self._fragment(
                    run_id,
                    "two",
                    {"test_a": ["torch/b.py"], "test_zero": []},
                ),
            )
            (fragment_dir / "worker.running").touch()

            incomplete = merge_td_tracer_fragments(output_path, run_id)
            self.assertFalse(incomplete["complete"])
            self.assertEqual(incomplete["running_participants"], ["worker"])

            (fragment_dir / "worker.running").unlink()
            complete = merge_td_tracer_fragments(output_path, run_id)
            self.assertTrue(complete["complete"])
            self.assertTrue(complete["usable"])
            self.assertNotIn("revision", complete)
            self.assertEqual(
                complete["environments"],
                [{"python": "test", "revision": "revision"}],
            )
            self.assertEqual(
                complete["coverage_by_test"],
                {
                    "test_a": ["torch/a.py", "torch/b.py"],
                    "test_zero": [],
                },
            )

    def test_bad_fragment_preserves_previous_output(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "td.json"
            run_id = "run"
            _atomic_write_json(output_path, {"old": True})
            bad_fragment = self._fragment(run_id, "bad", {})
            bad_fragment["schema_version"] += 1
            _atomic_write_json(_run_dir(output_path, run_id) / "bad.json", bad_fragment)

            with self.assertRaisesRegex(ValueError, "Unsupported TD tracer schema"):
                merge_td_tracer_fragments(output_path, run_id)
            with output_path.open(encoding="utf-8") as input_file:
                self.assertEqual(json.load(input_file), {"old": True})

    def test_outer_test_failure_marks_output_unusable(self):
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "td.json"
            run_id = "run"
            _atomic_write_json(
                _run_dir(output_path, run_id) / "participant.json",
                self._fragment(run_id, "participant", {}),
            )

            artifact = finalize_td_tracer(output_path, run_id, test_successful=False)
            remerged = merge_td_tracer_fragments(output_path, run_id)
            refinalized = finalize_td_tracer(output_path, run_id, test_successful=True)

        for output in (artifact, remerged, refinalized):
            self.assertFalse(output["complete"])
            self.assertFalse(output["successful"])
            self.assertFalse(output["usable"])

    def test_python_path_requires_canonical_root_containment(self):
        with TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            torch_dir = repo_root / "torch"
            torchgen_dir = repo_root / "torchgen"
            torch_dir.mkdir()
            torchgen_dir.mkdir()
            torch_file = torch_dir / "inside.py"
            torchgen_file = torchgen_dir / "outside.py"
            torch_file.write_text("", encoding="utf-8")
            torchgen_file.write_text("", encoding="utf-8")

            tracer = TDTracer.__new__(TDTracer)
            tracer._repo_root = repo_root
            tracer._python_roots = (torch_dir,)
            self.assertEqual(
                tracer._relative_python_path(str(torch_file)), "torch/inside.py"
            )
            self.assertIsNone(tracer._relative_python_path(str(torchgen_file)))
            self.assertIsNone(
                tracer._relative_python_path(f"{torch_file}:generated_code")
            )

    def test_merge_ci_shard_outputs(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            outputs = [
                self._shard_output("run-1-default-1", {"test_a": ["torch/a.py"]}),
                self._shard_output(
                    "run-1-default-2",
                    {
                        "test_a": ["torch/b.py"],
                        "test_zero": [],
                    },
                ),
            ]
            paths = []
            for index, output in enumerate(outputs):
                path = tmp_path / f"shard-{index}.json"
                _atomic_write_json(path, output)
                paths.append(path)

            merged = merge_td_tracer_outputs(
                paths,
                "ci-run",
                "run",
                1,
                {"default-1", "default-2"},
            )

        self.assertTrue(merged["usable"])
        self.assertEqual(merged["run_id"], "ci-run")
        self.assertEqual(
            merged["coverage_by_test"],
            {
                "test_a": ["torch/a.py", "torch/b.py"],
                "test_zero": [],
            },
        )
        self.assertEqual(
            [participant["participant_id"] for participant in merged["participants"]],
            ["run-1-default-1", "run-1-default-2"],
        )

    def test_merge_ci_shard_outputs_requires_every_shard(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "shard.json"
            _atomic_write_json(
                path,
                self._shard_output("run-1-default-1", {}),
            )
            merged = merge_td_tracer_outputs(
                [path],
                "ci-run",
                "run",
                1,
                {"default-1", "default-2"},
            )

        self.assertFalse(merged["complete"])
        self.assertFalse(merged["successful"])
        self.assertFalse(merged["usable"])

    def test_merge_ci_shard_outputs_unions_attempts_and_uses_latest_status(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = []
            outputs = [
                self._shard_output(
                    "run-1-default-1",
                    {"test": ["torch/old.py"]},
                    complete=False,
                    successful=False,
                ),
                self._shard_output("run-2-default-1", {"test": ["torch/new.py"]}),
                self._shard_output("run-1-default-2", {"test": ["torch/other.py"]}),
            ]
            for output in outputs:
                path = tmp_path / f"{output['run_id']}.json"
                _atomic_write_json(path, output)
                paths.append(path)

            merged = merge_td_tracer_outputs(
                paths,
                "run-2",
                "run",
                2,
                {"default-1", "default-2"},
            )

        self.assertTrue(merged["usable"])
        self.assertEqual(
            merged["coverage_by_test"],
            {"test": ["torch/new.py", "torch/old.py", "torch/other.py"]},
        )
        self.assertEqual(
            [participant["participant_id"] for participant in merged["participants"]],
            ["run-1-default-2", "run-2-default-1"],
        )

    def test_merge_ci_shard_outputs_prefer_newer_failure(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            old_path = tmp_path / "old.json"
            new_path = tmp_path / "new.json"
            _atomic_write_json(
                old_path,
                self._shard_output("run-1-default-1", {}),
            )
            _atomic_write_json(
                new_path,
                self._shard_output("run-2-default-1", {}, successful=False),
            )

            merged = merge_td_tracer_outputs(
                [old_path, new_path], "run-2", "run", 2, {"default-1"}
            )

        self.assertFalse(merged["successful"])
        self.assertFalse(merged["usable"])

    @parametrize(
        "source_run_id",
        ["other-1-default-1", "run-3-default-1", "run-2-unknown-1"],
    )
    def test_merge_ci_shard_outputs_rejects_unexpected_run_id(self, source_run_id):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "shard.json"
            _atomic_write_json(path, self._shard_output(source_run_id, {}))

            with self.assertRaisesRegex(ValueError, "Unexpected TD tracer run ID"):
                merge_td_tracer_outputs([path], "run-2", "run", 2, {"default-1"})

    def test_merge_ci_shard_outputs_rejects_duplicate_run_id(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths = [tmp_path / "one.json", tmp_path / "two.json"]
            for path in paths:
                _atomic_write_json(
                    path,
                    self._shard_output("run-1-default-1", {}),
                )

            with self.assertRaisesRegex(ValueError, "Duplicate TD tracer run ID"):
                merge_td_tracer_outputs(paths, "run-1", "run", 1, {"default-1"})

    def test_merge_ci_shard_outputs_rejects_malformed_input(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "shard.json"
            output = self._shard_output("shard", {})
            del output["environments"]
            _atomic_write_json(path, output)

            with self.assertRaisesRegex(ValueError, "environments"):
                merge_td_tracer_outputs([path], "ci-run", "run", 1, {"default-1"})

    def test_custom_pytest_requires_tracer_fragment(self):
        from run_test import _run_custom_pytest

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "td.json"
            env = {
                "PYTORCH_TD_TRACER_OUTPUT": str(output_path),
                "PYTORCH_TD_TRACER_RUN_ID": "run",
            }
            options = SimpleNamespace(additional_args=[])
            with patch("run_test.shell", return_value=0):
                return_code = _run_custom_pytest([], tmp_dir, options, env)

        self.assertEqual(return_code, 1)

    def test_doctest_handler_runs_pytest_with_tracer(self):
        from run_test import run_doctests

        options = SimpleNamespace(
            additional_args=["--td-tracer=/tmp/td.json"],
            td_tracer_enabled=True,
            verbose=0,
            xdoctest_command="all",
        )
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("run_test._run_custom_pytest", return_value=0) as run_pytest,
        ):
            return_code = run_doctests(
                SimpleNamespace(name="doctests"), str(REPO_ROOT / "test"), options
            )

        self.assertEqual(return_code, 0)
        command, test_directory, passed_options, env = run_pytest.call_args.args
        self.assertEqual(command[:4], [sys.executable, "-m", "pytest", "-p"])
        self.assertIn("--xdoctest-modules", command)
        self.assertEqual(test_directory, str(REPO_ROOT / "test"))
        self.assertIs(passed_options, options)
        self.assertEqual(env[TD_TRACER_CURRENT_TEST_ENV], "doctests")

    @parametrize("command", ["list", "torch.add"])
    def test_doctest_handler_rejects_command_with_tracer(self, command):
        from run_test import run_doctests

        options = SimpleNamespace(
            td_tracer_enabled=True,
            xdoctest_command=command,
        )
        with patch("run_test.print_to_stderr") as print_to_stderr:
            return_code = run_doctests(
                SimpleNamespace(name="doctests"), str(REPO_ROOT / "test"), options
            )

        self.assertEqual(return_code, 1)
        print_to_stderr.assert_called_once()

    def test_doctest_handler_keeps_native_runner_without_tracer(self):
        from run_test import run_doctests

        options = SimpleNamespace(
            td_tracer_enabled=False,
            verbose=0,
            xdoctest_command="list",
        )
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "xdoctest.runner.doctest_module", return_value={"action": "list"}
            ) as run_xdoctest,
        ):
            return_code = run_doctests(
                SimpleNamespace(name="doctests"), str(REPO_ROOT / "test"), options
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(run_xdoctest.call_args.kwargs["command"], "list")

    def test_autoload_handler_runs_pytest_with_tracer(self):
        from run_test import test_autoload_enable

        options = SimpleNamespace(
            additional_args=["--td-tracer=/tmp/td.json"],
            td_tracer_enabled=True,
        )
        test_module = SimpleNamespace(name="test_autoload_enable")
        with TemporaryDirectory() as install_directory:
            with (
                patch.dict(os.environ, {}, clear=False),
                patch(
                    "run_test.install_cpp_extensions",
                    return_value=(install_directory, 0),
                ),
                patch("run_test._run_custom_pytest", return_value=0) as run_pytest,
            ):
                return_code = test_autoload_enable(
                    test_module, str(REPO_ROOT / "test"), options
                )

        self.assertEqual(return_code, 0)
        command, test_directory, passed_options, env = run_pytest.call_args.args
        self.assertEqual(
            command,
            [sys.executable, "-m", "pytest", "-q", "test_autoload.py"],
        )
        self.assertEqual(test_directory, str(REPO_ROOT / "test"))
        self.assertIs(passed_options, options)
        self.assertEqual(env["TORCH_DEVICE_BACKEND_AUTOLOAD"], "1")
        self.assertEqual(env[TD_TRACER_CURRENT_TEST_ENV], "test_autoload_enable")
        self.assertIn(install_directory, env["PYTHONPATH"].split(os.pathsep))

    def test_openreg_handler_runs_pytest_with_tracer(self):
        from run_test import test_openreg

        options = SimpleNamespace(
            additional_args=["--td-tracer=/tmp/td.json"],
            td_tracer_enabled=True,
        )
        test_module = SimpleNamespace(name="test_openreg")
        tests_dir = (
            REPO_ROOT
            / "test/cpp_extensions/open_registration_extension/torch_openreg/tests"
        )
        with TemporaryDirectory() as install_directory:
            with (
                patch.dict(os.environ, {}, clear=False),
                patch(
                    "run_test.install_cpp_extensions",
                    return_value=(install_directory, 0),
                ),
                patch("run_test.os.path.isfile", return_value=False),
                patch("run_test._run_custom_pytest", return_value=0) as run_pytest,
            ):
                return_code = test_openreg(
                    test_module, str(REPO_ROOT / "test"), options
                )

        self.assertEqual(return_code, 0)
        command, test_directory, passed_options, env = run_pytest.call_args.args
        self.assertEqual(
            command,
            [sys.executable, "-m", "pytest", "-q", str(tests_dir)],
        )
        self.assertEqual(test_directory, str(REPO_ROOT / "test"))
        self.assertIs(passed_options, options)
        self.assertEqual(env[TD_TRACER_CURRENT_TEST_ENV], "test_openreg")
        self.assertIn(install_directory, env["PYTHONPATH"].split(os.pathsep))

    def test_openreg_handler_keeps_unittest_without_tracer(self):
        from run_test import test_openreg

        options = SimpleNamespace(td_tracer_enabled=False)
        tests_dir = (
            REPO_ROOT
            / "test/cpp_extensions/open_registration_extension/torch_openreg/tests"
        )
        with TemporaryDirectory() as install_directory:
            with (
                patch(
                    "run_test.install_cpp_extensions",
                    return_value=(install_directory, 0),
                ),
                patch("run_test.os.path.isfile", return_value=False),
                patch("run_test.shell", return_value=0) as run_unittest,
            ):
                return_code = test_openreg(
                    SimpleNamespace(name="test_openreg"),
                    str(REPO_ROOT / "test"),
                    options,
                )

        self.assertEqual(return_code, 0)
        self.assertEqual(
            run_unittest.call_args.args[0],
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(tests_dir),
                "-v",
            ],
        )


instantiate_parametrized_tests(TestTDTracerIntegration)
instantiate_parametrized_tests(TestTDTracerArtifacts)


if __name__ == "__main__":
    run_tests()

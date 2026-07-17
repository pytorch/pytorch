# Owner(s): ["module: tests"]

import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from td_tracer import (
    _atomic_write_json,
    _run_dir,
    merge_td_tracer_fragments,
    TD_TRACER_CURRENT_TEST_ENV,
    TD_TRACER_RUN_ID_ENV,
    TDTracer,
)

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(hasattr(sys, "monitoring"), "requires Python 3.12+")
class TestTDTracerIntegration(TestCase):
    def _write_project(
        self,
        project_dir: Path,
        files: dict[str, str],
        conftest_extra: str = "",
    ) -> None:
        conftest = f"""
from td_tracer import TD_TRACER_OPTION_NAME, td_tracer_arguments, TDTracer

def pytest_addoption(parser):
    td_tracer_arguments(parser)

def pytest_configure(config):
    if config.getoption(TD_TRACER_OPTION_NAME):
        config.pluginmanager.register(TDTracer(config), "pytesttraceplugin")

{conftest_extra}
"""
        (project_dir / "conftest.py").write_text(
            textwrap.dedent(conftest), encoding="utf-8"
        )
        for filename, contents in files.items():
            (project_dir / filename).write_text(
                textwrap.dedent(contents), encoding="utf-8"
            )

    def _subprocess_env(self, run_id: str | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env.pop(TD_TRACER_CURRENT_TEST_ENV, None)
        env.pop("PYTEST_XDIST_TESTRUNUID", None)
        if run_id is None:
            env.pop(TD_TRACER_RUN_ID_ENV, None)
        else:
            env[TD_TRACER_RUN_ID_ENV] = run_id
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
    ) -> dict:
        with TemporaryDirectory() as tmp_dir:
            project_dir = Path(tmp_dir)
            self._write_project(project_dir, files, conftest_extra)

            output_path = project_dir / "td.json"
            env = self._subprocess_env()
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


if __name__ == "__main__":
    run_tests()

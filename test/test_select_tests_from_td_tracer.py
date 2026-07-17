# Owner(s): ["module: ci"]

import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.testing import select_tests_from_td_tracer


sys.path.remove(str(REPO_ROOT))


class TestSelectTestsFromTDTracer(TestCase):
    def _git(self, repo: Path, *args: str) -> str:
        result = subprocess.run(
            [
                "git",
                "-c",
                "user.name=TD Test",
                "-c",
                "user.email=td-test@example.com",
                "-c",
                "commit.gpgSign=false",
                *args,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _write_trace(
        self,
        path: Path,
        coverage: object,
        *,
        schema_version: object = 4,
        usable: object = True,
        complete: object | None = None,
        successful: object | None = None,
        running_participants: object | None = None,
        environments: object | None = None,
        indent: int | None = None,
    ) -> None:
        status = usable if type(usable) is bool else True
        if complete is None:
            complete = status
        if successful is None:
            successful = status
        if running_participants is None:
            running_participants = []
        if environments is None:
            environments = [{"revision": "revision"}]
        with path.open("w", encoding="utf-8") as output:
            json.dump(
                {
                    "run_id": "run",
                    "complete": complete,
                    "successful": successful,
                    "running_participants": running_participants,
                    "participants": [],
                    "environments": environments,
                    "ignored": [{"nested": ["value", {"value": True}]}],
                    "coverage_by_test": coverage,
                    "usable": usable,
                    "schema_version": schema_version,
                },
                output,
                indent=indent,
            )

    def test_select_tests_with_details(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(
                path,
                {
                    "test/z.py::test_shared": ["torch/shared.py"],
                    "test/direct.py::test_direct": ["torch/shared.py"],
                    "zero-edge": [],
                },
                environments=[
                    {"revision": "revision-b"},
                    {"revision": None},
                    {"revision": "revision-a"},
                    {"revision": "revision-b"},
                ],
            )

            result = select_tests_from_td_tracer.select_tests_with_details(
                path,
                [
                    "./torch/shared.py",
                    "test\\direct.py",
                    "torch/missing.py",
                    "aten/a.cpp",
                    "torch/shared.py",
                ],
            )

        self.assertIsInstance(result, select_tests_from_td_tracer.SelectionResult)
        self.assertEqual(
            result.tests,
            ["test/direct.py::test_direct", "test/z.py::test_shared"],
        )
        self.assertEqual(
            result.matches_by_test,
            {
                "test/direct.py::test_direct": [
                    "test/direct.py",
                    "torch/shared.py",
                ],
                "test/z.py::test_shared": ["torch/shared.py"],
            },
        )
        self.assertEqual(
            result.affected,
            frozenset(
                {
                    "aten/a.cpp",
                    "test/direct.py",
                    "torch/missing.py",
                    "torch/shared.py",
                }
            ),
        )
        self.assertEqual(
            result.matched, frozenset({"test/direct.py", "torch/shared.py"})
        )
        self.assertEqual(result.unsupported, ["aten/a.cpp"])
        self.assertEqual(result.unmatched, ["torch/missing.py"])
        self.assertEqual(
            result.metadata,
            select_tests_from_td_tracer.TraceMetadata(
                schema_version=4,
                run_id="run",
                complete=True,
                successful=True,
                usable=True,
                revisions=("revision-a", "revision-b"),
            ),
        )

    def test_selects_dependency_and_direct_test_file_matches(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(
                path,
                {
                    "test/z.py::test_shared": ["torch/shared.py"],
                    "test/a.py::test_both": ["torch/shared.py", "torch/other.py"],
                    "test/direct.py::test_direct": [],
                    "logical-suite": ["torch/shared.py.bak"],
                    "zero-edge": [],
                },
                indent=2,
            )

            tests = select_tests_from_td_tracer.select_tests(
                path,
                ["./torch/shared.py", "test\\direct.py", "torch/shared.py"],
            )

        self.assertEqual(
            tests,
            [
                "test/a.py::test_both",
                "test/direct.py::test_direct",
                "test/z.py::test_shared",
            ],
        )

    def test_preserves_escaped_identifiers_and_paths(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(
                path,
                {'test/test_a.py::test_value["quoted"]': ['torch/quo"ted.py']},
            )

            tests = select_tests_from_td_tracer.select_tests(
                path, ['torch/quo"ted.py']
            )

        self.assertEqual(tests, ['test/test_a.py::test_value["quoted"]'])

    def test_matches_torch_relative_test_id(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(
                path,
                {"fx/tests/test_a.py::test_a": []},
            )

            tests = select_tests_from_td_tracer.select_tests(
                path, ["torch/fx/tests/test_a.py"]
            )

        self.assertEqual(tests, ["fx/tests/test_a.py::test_a"])

    def test_rejects_non_printable_test_id(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {"test/a.py::test_a\nforged": ["torch/a.py"]})

            with self.assertRaisesRegex(ValueError, "invalid test ID"):
                select_tests_from_td_tracer.select_tests(path, ["torch/a.py"])

    def test_preserves_long_test_id(self):
        test = f"test/a.py::test_a[{'x' * 4096}]"
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {test: ["torch/a.py"]})

            tests = select_tests_from_td_tracer.select_tests(path, ["torch/a.py"])

        self.assertEqual(tests, [test])

    def test_does_not_load_the_full_document(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {"test": ["torch/a.py"]})
            with mock.patch.object(
                select_tests_from_td_tracer.json,
                "load",
                side_effect=AssertionError("full JSON load"),
            ):
                result = select_tests_from_td_tracer.select_tests_with_details(
                    path, ["torch/a.py"]
                )

        self.assertEqual(result.tests, ["test"])

    def test_unusable_trace_still_returns_observed_matches(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {"test/a.py::test_a": ["torch/a.py"]}, usable=False)

            tests = select_tests_from_td_tracer.select_tests(path, ["torch/a.py"])
            result = select_tests_from_td_tracer.select_tests_with_details(
                path, ["torch/a.py"]
            )

        self.assertEqual(tests, ["test/a.py::test_a"])
        self.assertFalse(result.metadata.complete)
        self.assertFalse(result.metadata.successful)
        self.assertFalse(result.metadata.usable)

    def test_rejects_balanced_malformed_metadata(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {"test": ["torch/a.py"]})
            contents = path.read_text(encoding="utf-8")
            contents = contents.replace(
                '"ignored": [{"nested": ["value", {"value": true}]}]',
                '"ignored": [true false]',
            )
            path.write_text(contents, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid value"):
                select_tests_from_td_tracer.select_tests(path, ["torch/a.py"])

    def test_rejects_inconsistent_status(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(
                path,
                {},
                usable=True,
                complete=False,
                successful=True,
            )

            with self.assertRaisesRegex(ValueError, "Inconsistent TD tracer status"):
                select_tests_from_td_tracer.select_tests(path, ["torch/a.py"])

    def test_rejects_completed_trace_with_running_participants(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {}, running_participants=["worker"])

            with self.assertRaisesRegex(ValueError, "has running participants"):
                select_tests_from_td_tracer.select_tests(path, ["torch/a.py"])

    def test_changed_files_for_commit(self):
        with TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            self._git(repo, "init", "-q")
            (repo / "torch").mkdir()
            original = repo / "torch" / "original.py"
            original.write_text("original\n", encoding="utf-8")
            self._git(repo, "add", "--all")
            self._git(repo, "commit", "-qm", "root")
            root = self._git(repo, "rev-parse", "HEAD")
            primary_branch = self._git(repo, "branch", "--show-current")

            self.assertEqual(
                select_tests_from_td_tracer.changed_files_for_commit(root, repo),
                ["torch/original.py"],
            )

            self._git(repo, "mv", "torch/original.py", "torch/renamed.py")
            self._git(repo, "commit", "-qm", "rename")
            rename = self._git(repo, "rev-parse", "HEAD")
            self.assertEqual(
                select_tests_from_td_tracer.changed_files_for_commit(rename, repo),
                ["torch/original.py", "torch/renamed.py"],
            )

            self._git(repo, "checkout", "-qb", "side")
            (repo / "torch" / "side file.py").write_text("side\n", encoding="utf-8")
            self._git(repo, "add", "--all")
            self._git(repo, "commit", "-qm", "side")
            self._git(repo, "checkout", "-q", primary_branch)
            (repo / "torch" / "main.py").write_text("main\n", encoding="utf-8")
            self._git(repo, "add", "--all")
            self._git(repo, "commit", "-qm", "main")
            self._git(repo, "merge", "-q", "--no-ff", "-m", "merge", "side")
            merge = self._git(repo, "rev-parse", "HEAD")

            self.assertEqual(
                select_tests_from_td_tracer.changed_files_for_commit(merge, repo),
                ["torch/side file.py"],
            )

    def test_changed_files_rejects_invalid_revision(self):
        with TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            self._git(repo, "init", "-q")

            with self.assertRaisesRegex(ValueError, "must not start"):
                select_tests_from_td_tracer.changed_files_for_commit(
                    "--not-a-revision", repo
                )
            with self.assertRaisesRegex(ValueError, "Git command failed"):
                select_tests_from_td_tracer.changed_files_for_commit("missing", repo)

    @parametrize(
        "contents,error",
        [
            ("not JSON", "expected '{'"),
            ('{"schema_version": 3, "usable": true, "coverage_by_test": {}}', "schema"),
            (
                '{"schema_version": 4, "usable": true, "coverage_by_test": []}',
                "expected '{'",
            ),
            (
                '{"schema_version": 4, "usable": true, "coverage_by_test": {"test": [1]}}',
                "expected a string",
            ),
            ('{"schema_version": 4, "coverage_by_test": {}}', "usable"),
        ],
    )
    def test_rejects_invalid_trace(self, contents, error):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            path.write_text(contents, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, error):
                select_tests_from_td_tracer.select_tests(path, ["torch/a.py"])

    @parametrize(
        "affected", ["/torch/a.py", "../torch/a.py", "C:torch/a.py", "", "."]
    )
    def test_rejects_invalid_affected_file(self, affected):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {})

            with self.assertRaisesRegex(ValueError, "Affected file"):
                select_tests_from_td_tracer.select_tests(path, [affected])

    def test_cli_prints_sorted_tests_and_unusable_warning(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(
                path,
                {
                    "test/z.py::test_z": ["torch/a.py"],
                    "test/a.py::test_a": ["torch/a.py"],
                },
                usable=False,
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    select_tests_from_td_tracer,
                    "changed_files_for_commit",
                    return_value=["torch/a.py"],
                ) as changed_files,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                return_code = select_tests_from_td_tracer.main(
                    [str(path), "commit"]
                )

        self.assertEqual(return_code, 0)
        changed_files.assert_called_once_with("commit")
        self.assertEqual(
            stdout.getvalue(),
            "test/a.py::test_a\ntest/z.py::test_z\n",
        )
        self.assertIn("warning: TD tracer output is not usable", stderr.getvalue())
        self.assertIn("Selected 2 tests for 1 affected file", stderr.getvalue())

    def test_cli_warns_about_files_the_trace_cannot_cover(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            self._write_trace(path, {})
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    select_tests_from_td_tracer,
                    "changed_files_for_commit",
                    return_value=["aten/a.cpp", "torch/new.py"],
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                return_code = select_tests_from_td_tracer.main(
                    [str(path), "commit"]
                )

        self.assertEqual(return_code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("1 affected file is outside TD tracer coverage", stderr.getvalue())
        self.assertIn("aten/a.cpp", stderr.getvalue())
        self.assertIn("1 traceable affected file was not observed", stderr.getvalue())
        self.assertIn("torch/new.py", stderr.getvalue())

    def test_cli_does_not_print_partial_results_on_error(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trace.json"
            path.write_text(
                '{"schema_version": 4, "usable": true, '
                '"coverage_by_test": {"selected": ["torch/a.py"]}, '
                '"trailing": [}',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    select_tests_from_td_tracer,
                    "changed_files_for_commit",
                    return_value=["torch/a.py"],
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                return_code = select_tests_from_td_tracer.main(
                    [str(path), "commit"]
                )

        self.assertEqual(return_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("error: Invalid TD tracer JSON", stderr.getvalue())

    def test_cli_does_not_print_results_for_invalid_commit(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                select_tests_from_td_tracer,
                "changed_files_for_commit",
                side_effect=select_tests_from_td_tracer.TDSelectionError(
                    "invalid commit"
                ),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            return_code = select_tests_from_td_tracer.main(
                ["missing-trace.json", "bad-commit"]
            )

        self.assertEqual(return_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("error: invalid commit", stderr.getvalue())


instantiate_parametrized_tests(TestSelectTestsFromTDTracer)


if __name__ == "__main__":
    run_tests()

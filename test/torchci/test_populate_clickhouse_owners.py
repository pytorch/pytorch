# Owner(s): ["module: ci"]

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import call, MagicMock, patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.torchci import populate_clickhouse_owners


class TestClickHouseOwners(TestCase):
    repo_root = Path("/fake/pytorch")

    def test_owner_updates(self):
        stored_owners = {
            "test/unchanged.py": ["ci", "dynamo"],
            "test/changed.py": ["old"],
            "test/removed.py": ["ci"],
            "test/already_removed.py": [],
            "test/header_removed.py": ["ci"],
            "test/legacy.py": ["module: ci", "oncall: pt2"],
        }
        current_owners = {
            "test/unchanged.py": ["dynamo", "ci"],
            "test/changed.py": ["new"],
            "test/header_removed.py": [],
            "test/added.py": ["ci"],
            "test/legacy.py": ["ci"],
            "test/unowned.py": [],
        }
        owner_updates = populate_clickhouse_owners.get_owner_updates(
            "pytorch/pytorch", current_owners, stored_owners
        )
        self.assertEqual(
            owner_updates,
            [
                ("pytorch/pytorch", "test/added.py", ["ci"]),
                ("pytorch/pytorch", "test/changed.py", ["new"]),
                ("pytorch/pytorch", "test/header_removed.py", []),
                ("pytorch/pytorch", "test/legacy.py", ["ci"]),
                ("pytorch/pytorch", "test/removed.py", []),
                ("pytorch/pytorch", "test/unowned.py", []),
            ],
        )
        stored_owners.update({file: owners for _, file, owners in owner_updates})
        self.assertEqual(
            populate_clickhouse_owners.get_owner_updates(
                "pytorch/pytorch", current_owners, stored_owners
            ),
            [],
        )

    def test_database_required(self):
        environment = {
            "CLICKHOUSE_ENDPOINT": "https://localhost:8443",
            "CLICKHOUSE_USERNAME": "test-user",
            "CLICKHOUSE_PASSWORD": "test-password",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                populate_clickhouse_owners.subprocess, "check_output"
            ) as check_output,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            populate_clickhouse_owners.main([])
        self.assertEqual(error.exception.code, 2)
        check_output.assert_not_called()

    @parametrize("missing", [None, "ENDPOINT", "USERNAME", "PASSWORD"])
    @parametrize("empty", [False, True])
    def test_dry_run(self, missing, empty):
        current_owners = {"test/test_example.py": ["ci"]}
        environment = {}
        for name in ("ENDPOINT", "USERNAME", "PASSWORD"):
            if missing is not None and name != missing:
                environment[f"CLICKHOUSE_{name}"] = "test-value"
            elif empty:
                environment[f"CLICKHOUSE_{name}"] = ""
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch.object(
                populate_clickhouse_owners.subprocess,
                "check_output",
                side_effect=[f"{self.repo_root}\n", json.dumps(current_owners)],
            ) as check_output,
            patch.dict(sys.modules, {"clickhouse_connect": None}),
            patch.dict(os.environ, environment, clear=True),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            populate_clickhouse_owners.main(["--repo", "pytorch/example"])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue().splitlines(),
            [
                "Dry run: 1 test files found; database not updated (missing credentials).",
            ],
        )
        self.assertEqual(check_output.call_count, 2)
        command = [sys.executable, "scripts/codeowners/owners.py"]
        check_output.assert_has_calls(
            [
                call(["git", "rev-parse", "--show-toplevel"], text=True),
                call(command, cwd=str(self.repo_root), text=True),
            ],
        )

    def test_owner_collection_failure(self):
        environment = {
            "CLICKHOUSE_ENDPOINT": "https://localhost:8443",
            "CLICKHOUSE_USERNAME": "test-user",
            "CLICKHOUSE_PASSWORD": "test-password",
        }
        command = [sys.executable, "scripts/codeowners/owners.py"]
        failure = subprocess.CalledProcessError(1, command)
        clickhouse = MagicMock()
        with (
            patch.object(
                populate_clickhouse_owners.subprocess,
                "check_output",
                side_effect=[f"{self.repo_root}\n", failure],
            ),
            patch.dict(sys.modules, {"clickhouse_connect": clickhouse}),
            patch.dict(os.environ, environment, clear=True),
            self.assertRaisesRegex(
                subprocess.CalledProcessError, "non-zero exit status 1"
            ) as error,
        ):
            populate_clickhouse_owners.main(["--database", "fortesting"])
        self.assertIs(error.exception, failure)
        clickhouse.get_client.assert_not_called()

    def test_dry_run_uses_shared_owners(self):
        repo_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        for name in ("ENDPOINT", "USERNAME", "PASSWORD"):
            environment.pop(f"CLICKHOUSE_{name}", None)
        environment.pop("PYTHONPATH", None)
        expected_owners = json.loads(
            subprocess.check_output(
                [sys.executable, "-S", str(repo_root / "scripts/codeowners/owners.py")],
                cwd=repo_root,
                env=environment,
                text=True,
            )
        )
        script = Path(populate_clickhouse_owners.__file__).resolve()
        result = subprocess.run(
            [sys.executable, "-I", "-S", str(script), "--repo", "pytorch/example"],
            cwd=repo_root / "test",
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr.splitlines(),
            [
                f"Dry run: {len(expected_owners)} test files found; "
                "database not updated (missing credentials).",
            ],
        )

    @parametrize("database", ["fortesting", "ci"])
    @parametrize("unchanged", [False, True])
    def test_upload(self, database, unchanged):
        current_owners = {"test/test_example.py": ["ci"]}
        clickhouse = MagicMock()
        client = clickhouse.get_client.return_value.__enter__.return_value
        client.query.return_value.result_rows = (
            [("test/test_example.py", ["ci"])] if unchanged else []
        )
        environment = {
            "CLICKHOUSE_ENDPOINT": "https://localhost:8443",
            "CLICKHOUSE_USERNAME": "test-user",
            "CLICKHOUSE_PASSWORD": "test-password",
        }
        repo = "pytorch/fork'quoted"
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch.object(
                populate_clickhouse_owners.subprocess,
                "check_output",
                side_effect=[f"{self.repo_root}\n", json.dumps(current_owners)],
            ),
            patch.dict(sys.modules, {"clickhouse_connect": clickhouse}),
            patch.dict(os.environ, environment, clear=True),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            populate_clickhouse_owners.main(["--repo", repo, "--database", database])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(clickhouse.get_client.call_args.kwargs["database"], database)
        client.query.assert_called_once_with(
            "SELECT file, owners FROM owners FINAL WHERE repo = {repo:String}",
            parameters={"repo": repo},
        )
        if unchanged:
            client.insert.assert_not_called()
        else:
            client.insert.assert_called_once_with(
                "owners",
                [(repo, "test/test_example.py", ["ci"])],
                column_names=["repo", "file", "owners"],
            )
        clickhouse.get_client.return_value.__exit__.assert_called_once()
        self.assertEqual(
            stderr.getvalue().splitlines(),
            [
                f"Found 1 test files; updated {0 if unchanged else 1} files in {database}.owners",
            ],
        )


instantiate_parametrized_tests(TestClickHouseOwners)


if __name__ == "__main__":
    run_tests()

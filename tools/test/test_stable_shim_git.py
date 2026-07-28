import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

from tools.linter.adapters import _stable_shim_utils as shim_utils


class TestStableShimGit(TestCase):
    def _git(
        self,
        cwd: Path,
        *args: str,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=check,
            env=env,
            capture_output=True,
            text=True,
        )

    def _make_remote(self, root: Path) -> tuple[Path, str]:
        source = root / "source"
        remote = root / "remote.git"
        source.mkdir()
        self._git(source, "init", "-q")
        self._git(source, "config", "user.name", "Stable Shim Test")
        self._git(source, "config", "user.email", "stable-shim@example.com")

        (source / "diff.txt").write_text("base diff\n")
        (source / "read.txt").write_text("base read\n")
        self._git(source, "add", ".")
        self._git(source, "commit", "-qm", "base")
        self._git(source, "branch", "-M", "main")
        base = self._git(source, "rev-parse", "HEAD").stdout.strip()

        self._git(source, "switch", "-qc", "feature")
        (source / "diff.txt").write_text("feature diff\n")
        (source / "read.txt").write_text("feature read\n")
        (source / "new.txt").write_text("new\n")
        self._git(source, "add", ".")
        self._git(source, "commit", "-qm", "feature")

        self._git(root, "init", "--bare", "-q", str(remote))
        self._git(remote, "config", "uploadpack.allowFilter", "true")
        self._git(source, "remote", "add", "origin", remote.as_uri())
        self._git(source, "push", "-q", "origin", "main", "feature")
        self._git(remote, "symbolic-ref", "HEAD", "refs/heads/feature")
        return remote, base

    def _clone(
        self, root: Path, remote: Path, filter_spec: str | None = None
    ) -> Path:
        clone = root / "clone"
        args = ["clone", "-q"]
        if filter_spec is not None:
            args.append(f"--filter={filter_spec}")
        args.extend([remote.as_uri(), str(clone)])
        self._git(root, *args)
        return clone

    def _refs(self, repo: Path) -> str:
        return self._git(
            repo, "for-each-ref", "--format=%(refname) %(objectname)"
        ).stdout

    @parametrize("filter_spec", ["blob:none", "tree:0"])
    def test_partial_clone_hydration(self, filter_spec: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            remote, base = self._make_remote(root)
            clone = self._clone(root, remote, filter_spec)
            self.assertEqual(
                self._git(clone, "rev-parse", "refs/remotes/origin/main")
                .stdout.strip(),
                base,
            )

            real_run = subprocess.run
            merge_base_calls: list[list[str]] = []

            def record_merge_base(*args, **kwargs):
                merge_base_calls.append(args[0])
                return real_run(*args, **kwargs)

            with patch.object(shim_utils, "REPO_ROOT", clone):
                shim_utils.merge_base_with_main.cache_clear()
                try:
                    with patch.object(
                        shim_utils.subprocess,
                        "run",
                        side_effect=record_merge_base,
                    ):
                        self.assertEqual(shim_utils.merge_base_with_main(), base)
                finally:
                    shim_utils.merge_base_with_main.cache_clear()
            self.assertEqual(
                merge_base_calls,
                [["git", "merge-base", "HEAD", "refs/remotes/origin/main"]],
            )

            no_lazy = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
            local_diff = self._git(
                clone,
                "diff",
                f"{base}..HEAD",
                "--",
                "diff.txt",
                check=False,
                env=no_lazy,
            )
            self.assertNotEqual(local_diff.returncode, 0)

            refs_before = self._refs(clone)
            helper_calls: list[list[str]] = []

            def record_helper(*args, **kwargs):
                helper_calls.append(args[0])
                return real_run(*args, **kwargs)

            with patch.dict(os.environ, {"GIT_NO_LAZY_FETCH": "1"}):
                with patch.object(
                    shim_utils.subprocess,
                    "run",
                    side_effect=record_helper,
                ):
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        reads = list(
                            pool.map(
                                lambda _: shim_utils.read_file_at_revision(
                                    base, "read.txt", cwd=clone
                                ),
                                range(8),
                            )
                        )
                    diff = shim_utils.run_git_object_command(
                        ["diff", f"{base}..HEAD", "--", "diff.txt"], cwd=clone
                    )
                    missing = shim_utils.read_file_at_revision(
                        base, "new.txt", cwd=clone
                    )

            self.assertEqual(reads, ["base read\n"] * 8)
            self.assertIn("-base diff", diff.stdout)
            self.assertEqual(missing, None)
            self.assertEqual(self._refs(clone), refs_before)
            self.assertFalse(
                any(command[:2] == ["git", "fetch"] for command in helper_calls)
            )

            offline_diff = self._git(
                clone,
                "diff",
                f"{base}..HEAD",
                "--",
                "diff.txt",
                check=False,
                env=no_lazy,
            )
            self.assertEqual(offline_diff.returncode, 0, offline_diff.stderr)

    def test_full_clone_does_not_access_remote(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            remote, base = self._make_remote(root)
            clone = self._clone(root, remote)
            self._git(
                clone,
                "remote",
                "set-url",
                "origin",
                (root / "unreachable.git").as_uri(),
            )

            diff = shim_utils.run_git_object_command(
                ["diff", f"{base}..HEAD", "--", "diff.txt"], cwd=clone
            )
            contents = shim_utils.read_file_at_revision(
                base, "read.txt", cwd=clone
            )
            self.assertIn("-base diff", diff.stdout)
            self.assertEqual(contents, "base read\n")

    def test_merge_base_error_explains_how_to_fetch_main(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self._git(repo, "init", "-q")
            self._git(repo, "config", "user.name", "Stable Shim Test")
            self._git(repo, "config", "user.email", "stable-shim@example.com")
            (repo / "file.txt").write_text("contents\n")
            self._git(repo, "add", "file.txt")
            self._git(repo, "commit", "-qm", "initial")

            with patch.object(shim_utils, "REPO_ROOT", repo):
                shim_utils.merge_base_with_main.cache_clear()
                try:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "git fetch origin main:refs/remotes/origin/main",
                    ):
                        shim_utils.merge_base_with_main()
                finally:
                    shim_utils.merge_base_with_main.cache_clear()

    def test_historical_read_preserves_git_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self._git(repo, "init", "-q")
            with patch.object(shim_utils.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "invalid-revision"):
                    shim_utils.read_file_at_revision(
                        "invalid-revision", "missing.txt", cwd=repo
                    )


instantiate_parametrized_tests(TestStableShimGit)


if __name__ == "__main__":
    run_tests()

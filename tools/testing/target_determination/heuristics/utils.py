import os
import subprocess
from pathlib import Path
from typing import List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent


def python_test_file_to_test_name(tests: Set[str]) -> Set[str]:
    prefix = f"test{os.path.sep}"
    valid_tests = {f for f in tests if f.startswith(prefix) and f.endswith(".py")}
    valid_tests = {f[len(prefix) : -len(".py")] for f in valid_tests}

    return valid_tests


def query_changed_files() -> List[str]:
    default_branch = f"origin/{os.environ.get('GIT_DEFAULT_BRANCH', 'main')}"
    merge_base = (
        subprocess.check_output(["git", "merge-base", default_branch, "HEAD"])
        .decode()
        .strip()
    )

    head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

    base_commit = merge_base
    if base_commit == head:
        # We are on the default branch, so check for changes since the last commit
        base_commit = "HEAD^"

    proc = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"], capture_output=True
    )

    if proc.returncode != 0:
        raise RuntimeError("Unable to get changed files")

    lines = proc.stdout.decode().strip().split("\n")
    lines = [line.strip() for line in lines]
    return lines

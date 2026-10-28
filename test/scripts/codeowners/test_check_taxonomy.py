# Owner(s): ["module: ci"]

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/codeowners"))
from check_taxonomy import find_ownership_errors, parse_codeowners


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/codeowners/check_taxonomy.py"


def parse(codeowners: str):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "CODEOWNERS")
        path.write_text(codeowners)
        return parse_codeowners(path)


def ownership_errors(codeowners: str) -> dict[str, list[str]]:
    return {
        title: entries
        for title, entries in find_ownership_errors(parse(codeowners)).items()
        if entries
    }


def run_lint(codeowners: str, *args: str) -> subprocess.CompletedProcess:
    """Run check_taxonomy.py in a throwaway repo whose only committed file is CODEOWNERS."""
    with tempfile.TemporaryDirectory() as repo:
        Path(repo, "CODEOWNERS").write_text(codeowners)
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
        for command in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
            subprocess.check_output(["git", *identity, *command], cwd=repo)
        return subprocess.run(
            [sys.executable, "-S", str(SCRIPT), *args],
            cwd=repo,
            capture_output=True,
            text=True,
        )


LINT_HEADER = """\
# Review-surface hierarchy:
# - Protected ownership: protected_codeownership
#
# Grouped review-surface definitions:
#
# [protected_codeownership]
"""


class TestFindOwnershipErrors(TestCase):
    def test_valid_owner_lines(self):
        codeowners = """\
# Grouped review-surface definitions:
#
# [compiler_dynamo]
# /torch/_dynamo/unlabeled.py
# label: module: dynamo
# owner: @alice @bob
# /torch/_dynamo/
/torch/_dynamo/eval_frame.py @Alice
# label: module: dynamo, module: tvm
# owner: none
# /torch/_dynamo/backends/tvm.py
#
# [core_tests]
# label: module: dynamo
# owner: @Bob @alice
/test/dynamo/ @bob
"""
        self.assertEqual(ownership_errors(codeowners), {})

    def test_each_rule(self):
        codeowners = """\
# Grouped review-surface definitions:
#
# [compiler_dynamo]
# label: module: dynamo
/torch/_dynamo/ @alice
# label: module: inductor
# owner: @alice
# /torch/_inductor/
# owner: @bob
/torch/_inductor/codegen/ @carol
# label: module: fx
# owner: not-a-handle
# /torch/fx/
#
# [core_tests]
# label: module: inductor
# owner: @dave
# /test/inductor/
"""
        self.assertEqual(
            ownership_errors(codeowners),
            {
                "Label lines without an owner line": [
                    "CODEOWNERS:4 # label: module: dynamo"
                ],
                "Misplaced owner lines": ["CODEOWNERS:9 # owner: @bob"],
                "Invalid owner handles": ["CODEOWNERS:12 # owner: not-a-handle"],
                # The misplaced @bob line is reported above, not compared.
                "Label lists with several owner lists": [
                    "module: inductor: @alice | @dave"
                ],
            },
        )

    def test_owner_none_and_blocks_without_paths(self):
        # `none` is compared like any other owner list, and a label block with no
        # paths under it is still compared.
        codeowners = """\
# Grouped review-surface definitions:
#
# [compiler_dynamo]
# label: module: dynamo
# owner: none
# /torch/_dynamo/
# label: module: fx
# owner: @alice
#
# [core_tests]
# label: module: dynamo
# owner: @alice
# /test/dynamo/
# label: module: fx
# owner: @bob
# /test/fx/
"""
        self.assertEqual(
            ownership_errors(codeowners),
            {
                "Label lists with several owner lists": [
                    "module: dynamo: none | @alice",
                    "module: fx: @alice | @bob",
                ]
            },
        )

    def test_owners_reset_at_section(self):
        parsed = parse(
            """\
# Grouped review-surface definitions:
#
# [compiler_dynamo]
# label: module: dynamo
# owner: @alice
/torch/_dynamo/ @alice
#
# [core_tests]
/test/dynamo/ @bob
"""
        )
        unlabeled = parsed.patterns[1]
        self.assertEqual((unlabeled.labels, unlabeled.owners), ((), None))


class TestLintWiring(TestCase):
    def test_ownership_errors_fail_the_lint(self):
        codeowners = LINT_HEADER + "# label: module: ci\n/CODEOWNERS @alice\n#\n"
        result = run_lint(codeowners)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Label lines without an owner line (1):", result.stdout)

        result = run_lint(codeowners, "--lintrunner")
        self.assertEqual(result.returncode, 0, result.stderr)
        message = json.loads(result.stdout)
        self.assertEqual(message["code"], "CODEOWNERS_TAXONOMY")
        self.assertIn("1 label lines without an owner line", message["description"])

    def test_clean_file_passes(self):
        codeowners = (
            LINT_HEADER
            + "# label: module: ci\n# owner: @alice\n/CODEOWNERS @alice\n#\n"
        )
        result = run_lint(codeowners)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(run_lint(codeowners, "--lintrunner").stdout, "")


if __name__ == "__main__":
    run_tests()

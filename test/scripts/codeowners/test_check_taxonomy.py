# Owner(s): ["module: ci"]

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/codeowners"))
import check_taxonomy
from check_taxonomy import find_ownership_errors, parse_codeowners


def parse(codeowners: str):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "CODEOWNERS")
        path.write_text(codeowners)
        return parse_codeowners(path)


def ownership_errors(codeowners: str, *, exceptions=None) -> dict[str, list[str]]:
    errors = find_ownership_errors(
        parse(codeowners), reviewers_not_owners=exceptions or {}
    )
    return {title: entries for title, entries in errors.items() if entries}


def run_lint(codeowners: str, *args: str) -> tuple[int, str]:
    """Run the lint's main() in a throwaway repo whose only committed file is CODEOWNERS.

    The repo's REVIEWERS_NOT_OWNERS name paths that don't exist here, so the lint runs
    without exceptions.
    """
    with tempfile.TemporaryDirectory() as repo:
        Path(repo, "CODEOWNERS").write_text(codeowners)
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
        for command in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
            subprocess.check_output(["git", *identity, *command], cwd=repo)
        stdout = io.StringIO()
        previous = os.getcwd()
        os.chdir(repo)
        try:
            with (
                mock.patch.object(check_taxonomy, "REVIEWERS_NOT_OWNERS", {}),
                mock.patch.object(sys, "argv", ["check_taxonomy.py", *args]),
                contextlib.redirect_stdout(stdout),
            ):
                returncode = check_taxonomy.main()
        finally:
            os.chdir(previous)
        return returncode, stdout.getvalue()


LINT_HEADER = """\
# Review-surface hierarchy:
# - Protected ownership: protected_codeownership
#
# Grouped review-surface definitions:
#
# [protected_codeownership]
"""


class TestFindOwnershipErrors(TestCase):
    hw_classification = HardwareClassification.GENERIC

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
# label: module: dynamo, module: testing
# owner: @Bob @alice
/test/dynamo/ @bob
"""
        self.assertEqual(ownership_errors(codeowners), {})

    def test_invalid_owner_handles(self):
        codeowners = """\
# Grouped review-surface definitions:
#
# [compiler_dynamo]
# label: module: fx
# owner: not-a-handle
# /torch/fx/
"""
        self.assertEqual(
            ownership_errors(codeowners),
            {"Invalid owner handles": ["CODEOWNERS:5 # owner: not-a-handle"]},
        )

    def test_reviewers_must_be_owners(self):
        # A correctly placed owner line that misses a reviewer, and `none` with an
        # active rule, are reported; rules above a section's first label are not.
        codeowners = """\
# Grouped review-surface definitions:
#
# [compiler_dynamo]
/torch/_dynamo/unlabeled.py @carol
# label: module: dynamo
# owner: @alice
/torch/_dynamo/ @alice @bob
# label: module: fx
# owner: none
/torch/fx/ @dave
"""
        self.assertEqual(
            ownership_errors(codeowners),
            {
                "Reviewers missing from owner lists": [
                    "CODEOWNERS:7 /torch/_dynamo/: @bob not on the block's owner line",
                    "CODEOWNERS:10 /torch/fx/: @dave not on the block's owner line",
                ]
            },
        )

    def test_reviewer_exceptions(self):
        codeowners = """\
# Grouped review-surface definitions:
#
# [compiler_dynamo]
# label: module: dynamo
# owner: @alice
/torch/_dynamo/ @alice @bob
/torch/_dynamo/eval_frame.py @alice
"""
        exceptions = {
            "torch/_dynamo/": frozenset({"@Bob"}),
            "torch/_dynamo/eval_frame.py": frozenset({"@alice"}),
            "torch/removed.py": frozenset({"@carol"}),
        }
        # @bob is allowed even though the exception spells it @Bob. The eval_frame.py
        # exception is unneeded since @alice is an owner, and the removed rule's
        # exception is reported so it can't later exempt a new rule.
        self.assertEqual(
            ownership_errors(codeowners, exceptions=exceptions),
            {
                "Unneeded reviewer exceptions": [
                    "/torch/_dynamo/eval_frame.py: @alice",
                    "/torch/removed.py: @carol",
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
    hw_classification = HardwareClassification.GENERIC

    def test_ownership_errors_fail_the_lint(self):
        codeowners = (
            LINT_HEADER + "# label: module: ci\n# owner: @alice\n/CODEOWNERS @bob\n#\n"
        )
        returncode, stdout = run_lint(codeowners)
        self.assertEqual(returncode, 1, stdout)
        self.assertIn("Reviewers missing from owner lists (1):", stdout)

        returncode, stdout = run_lint(codeowners, "--lintrunner")
        self.assertEqual(returncode, 0, stdout)
        message = json.loads(stdout)
        self.assertEqual(message["code"], "CODEOWNERS_TAXONOMY")
        self.assertIn("1 reviewers missing from owner lists", message["description"])

    def test_clean_file_passes(self):
        codeowners = (
            LINT_HEADER
            + "# label: module: ci\n# owner: @alice\n/CODEOWNERS @alice\n#\n"
        )
        returncode, stdout = run_lint(codeowners)
        self.assertEqual(returncode, 0, stdout)
        self.assertEqual(run_lint(codeowners, "--lintrunner"), (0, ""))


if __name__ == "__main__":
    run_tests()

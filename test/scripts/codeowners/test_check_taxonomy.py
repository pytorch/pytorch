# Owner(s): ["module: ci"]

import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/codeowners"))
from check_taxonomy import find_ownership_errors, parse_patterns


def ownership_errors(codeowners: str) -> dict[str, list[str]]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "CODEOWNERS")
        path.write_text(codeowners)
        patterns, _ = parse_patterns(path)
        return {
            title: entries
            for title, entries in find_ownership_errors(path, patterns).items()
            if entries
        }


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
                "Label lists with several owner lists": [
                    "module: inductor: @alice | @bob | @dave"
                ],
            },
        )


if __name__ == "__main__":
    run_tests()

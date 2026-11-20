# Owner(s): ["module: ci"]

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/codeowners/generate_codeowners.py"
)
sys.path.insert(0, str(SCRIPT.parent))
from check_taxonomy import CODEOWNERS_PATH
from generate_codeowners import load_ownership, render_codeowners, SECTION_MARKER


OWNERSHIP = """\
planned_labels: ["module: tvm"]
sections:
  - compiler_dynamo
  - python_nn_api
  - protected_codeownership
section_notes:
  protected_codeownership:
    - "# Keep these rules last."
ownership:
  - name: linear_cross_entropy
    labels: ["module: linear cross entropy"]
    owners: [pearu]
    section: python_nn_api
    paths:
      - /torch/nn/modules/linear_cross_entropy.py: [pearu]
  - name: dynamo
    labels: ["module: dynamo"]
    owners: [alice, carol]
    section: compiler_dynamo
    paths:
      - /torch/_dynamo/
      - /torch/_dynamo/eval_frame.py: [alice, carol]
  - name: tvm_dynamo
    labels: ["module: dynamo", "module: tvm"]
    owners: []
    section: compiler_dynamo
    paths:
      - /torch/_dynamo/backends/tvm.py
  - name: nn
    labels: ["module: nn"]
    owners: [bob]
    section: python_nn_api
    paths:
      - /torch/nn/: [bob]
  - name: compiler_dynamo
    owners: []
    section: compiler_dynamo
    paths:
      - /torch/_dynamo_unlabeled.py
  - name: protected_codeownership
    owners: []
    section: protected_codeownership
    paths:
      - /CODEOWNERS: [albanD]
"""


def render(ownership: str) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "OWNERSHIP.yaml")
        path.write_text(ownership)
        return render_codeowners(load_ownership(path))


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"expected exactly one {old!r} in the fixture")
    return text.replace(old, new)


class TestGenerateCodeowners(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_render(self):
        text = render(OWNERSHIP)
        header, sections = text.split(SECTION_MARKER + "\n")
        self.assertIn("#   them until they exist): module: tvm\n", header)
        self.assertIn(
            "# - Sections: compiler_dynamo, python_nn_api, protected_codeownership\n",
            header,
        )
        # Paths without a label group come first in their section. In python_nn_api
        # the nn block (/torch/nn/) comes before the linear_cross_entropy block,
        # whose path is inside it, even though OWNERSHIP.yaml lists nn later.
        self.assertEqual(
            sections,
            """\
#
# [compiler_dynamo]
# /torch/_dynamo_unlabeled.py
# label: module: dynamo
# owner: @alice @carol
# /torch/_dynamo/
/torch/_dynamo/eval_frame.py @alice @carol
# label: module: dynamo, module: tvm
# owner: none
# /torch/_dynamo/backends/tvm.py
#
# [python_nn_api]
# label: module: nn
# owner: @bob
/torch/nn/ @bob
# label: module: linear cross entropy
# owner: @pearu
/torch/nn/modules/linear_cross_entropy.py @pearu
#
# [protected_codeownership]
# Keep these rules last.
/CODEOWNERS @albanD
#
""",
        )

    def test_no_planned_labels(self):
        text = render(replace_once(OWNERSHIP, '["module: tvm"]', "[]"))
        self.assertNotIn("Labels planned", text)

    def test_invalid_ownership(self):
        cases = {
            "every ownership entry needs a `name`": ("  - name: nn\n", "  - \n"),
            "duplicate OWNERSHIP.yaml names: dynamo": ("name: nn\n", "name: dynamo\n"),
            "section 'python_api' is not in `sections`": (
                "owners: [bob]\n    section: python_nn_api",
                "owners: [bob]\n    section: python_api",
            ),
            "must be a list of non-empty strings": (
                "/torch/nn/: [bob]",
                "/torch/nn/: bob",
            ),
            "a group without labels can't have owners": (
                "  - name: compiler_dynamo\n    owners: []",
                "  - name: compiler_dynamo\n    owners: [carol]",
            ),
            "'dynamo' and 'tvm_dynamo' have the same labels": (
                '["module: dynamo", "module: tvm"]',
                '["module: dynamo"]',
            ),
            "unknown keys \\['owner'\\]": (
                "owners: [pearu]",
                "owners: [pearu]\n    owner: pearu",
            ),
            "not valid YAML": ("planned_labels:", "planned_labels: [\n"),
            "'nn' owners: list people instead of teams \\['pytorch/nn-team'\\]": (
                "owners: [bob]",
                "owners: [bob, pytorch/nn-team]",
            ),
            "/torch/nn/: list GitHub logins instead of emails \\['bob@example.com'\\]": (
                "/torch/nn/: [bob]",
                "/torch/nn/: [bob@example.com]",
            ),
            "/torch/nn/: list people instead of teams \\['pytorch/nn-team'\\]": (
                "/torch/nn/: [bob]",
                "/torch/nn/: [bob, pytorch/nn-team]",
            ),
            "sits inside /torch/_dynamo/ in 'dynamo'": (
                "/torch/_dynamo_unlabeled.py",
                "/torch/_dynamo/unlabeled.py",
            ),
            "section_notes for sections not in `sections`": (
                "  protected_codeownership:\n    - ",
                "  protected:\n    - ",
            ),
        }
        for message, (old, new) in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    render(replace_once(OWNERSHIP, old, new))

    def test_one_of_owners_lists_the_owners_on_every_path(self):
        ownership = replace_once(
            OWNERSHIP,
            "    owners: [bob]\n    section: python_nn_api\n    paths:\n      - /torch/nn/: [bob]\n",
            "    owners: [bob, dave]\n    section: python_nn_api\n    review_request: one_of_owners\n"
            "    paths:\n      - /torch/nn/\n",
        )
        self.assertIn("# owner: @bob @dave\n/torch/nn/ @bob @dave\n", render(ownership))

    def test_invalid_one_of_owners(self):
        cases = {
            "review_request must be one of": "review_request: everyone",
            "with one_of_owners, write paths as bare strings": "review_request: one_of_owners",
        }
        for message, line in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    render(replace_once(OWNERSHIP, "owners: [bob]\n", f"owners: [bob]\n    {line}\n"))
        ownership = replace_once(
            OWNERSHIP,
            "  - name: tvm_dynamo\n",
            "  - name: tvm_dynamo\n    review_request: one_of_owners\n",
        )
        with self.assertRaisesRegex(ValueError, "one_of_owners needs owners"):
            render(ownership)

    def test_groups_nesting_both_ways(self):
        ownership = replace_once(
            OWNERSHIP,
            "      - /torch/nn/: [bob]\n",
            "      - /torch/nn/: [bob]\n      - /torch/nn/modules/x/: [bob]\n",
        )
        ownership = replace_once(
            ownership,
            "      - /torch/nn/modules/linear_cross_entropy.py: [pearu]\n",
            "      - /torch/nn/modules/: [pearu]\n",
        )
        with self.assertRaisesRegex(ValueError, "nest inside each other both ways"):
            render(ownership)


def run_generator(*, ownership: str, codeowners: str, args: list[str]):
    """Run the generator in a throwaway repo; return the result and final CODEOWNERS."""
    with tempfile.TemporaryDirectory() as repo:
        Path(repo, "OWNERSHIP.yaml").write_text(ownership)
        Path(repo, CODEOWNERS_PATH).parent.mkdir(parents=True)
        Path(repo, CODEOWNERS_PATH).write_text(codeowners)
        subprocess.check_output(["git", "init", "-q"], cwd=repo)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        return result, Path(repo, CODEOWNERS_PATH).read_text()


class TestGeneratorModes(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_up_to_date(self):
        generated = render(OWNERSHIP)
        result, _ = run_generator(
            ownership=OWNERSHIP, codeowners=generated, args=["--check"]
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result, _ = run_generator(
            ownership=OWNERSHIP, codeowners=generated, args=["--lintrunner"]
        )
        self.assertEqual((result.returncode, result.stdout), (0, ""))

    def test_stale_codeowners(self):
        generated = render(OWNERSHIP)
        result, _ = run_generator(
            ownership=OWNERSHIP, codeowners="stale\n", args=["--check"]
        )
        self.assertEqual(result.returncode, 1)
        result, _ = run_generator(
            ownership=OWNERSHIP, codeowners="stale\n", args=["--lintrunner"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        message = json.loads(result.stdout)
        self.assertEqual(
            (
                message["path"],
                message["code"],
                message["original"],
                message["replacement"],
            ),
            (CODEOWNERS_PATH, "CODEOWNERS_GENERATED", "stale\n", generated),
        )
        result, codeowners = run_generator(
            ownership=OWNERSHIP, codeowners="stale\n", args=[]
        )
        self.assertEqual((result.returncode, codeowners), (0, generated))

    def test_invalid_ownership(self):
        invalid = replace_once(OWNERSHIP, "name: nn\n", "name: dynamo\n")
        result, _ = run_generator(ownership=invalid, codeowners="x\n", args=["--check"])
        self.assertEqual(result.returncode, 1)
        self.assertIn("duplicate OWNERSHIP.yaml names", result.stderr)
        result, _ = run_generator(
            ownership=invalid, codeowners="x\n", args=["--lintrunner"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        message = json.loads(result.stdout)
        self.assertEqual(
            (message["path"], message["replacement"]), ("OWNERSHIP.yaml", None)
        )
        self.assertIn("duplicate OWNERSHIP.yaml names", message["description"])


if __name__ == "__main__":
    run_tests()

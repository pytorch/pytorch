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
from generate_codeowners import (
    LABEL_RULES_BEGIN,
    LABEL_RULES_END,
    load_ownership,
    OwnershipFile,
    render_codeowners,
    SECTION_MARKER,
    splice_label_rules,
)


KNOWN_LABELS = {
    "module: dynamo",
    "module: nn",
    "module: linear cross entropy",
    "oncall: pt2",
}


OWNERSHIP = """\
planned_labels: ["module: tvm"]
label_parents:
  "module: dynamo": "oncall: pt2"
  "module: tvm": "module: dynamo"
  "module: linear cross entropy": "module: nn"
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
    owners: [alice, pytorch/dev-infra]
    section: compiler_dynamo
    paths:
      - /torch/_dynamo/
      - /torch/_dynamo/eval_frame.py: [alice, someone@example.com]
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

LABEL_RULES = f"""\
- any:
  - "good first issue"
  then:
  - "actionable"
{LABEL_RULES_BEGIN}
- stale
{LABEL_RULES_END}
"""


def load(ownership: str) -> OwnershipFile:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory, "OWNERSHIP.yaml")
        path.write_text(ownership)
        return load_ownership(path, KNOWN_LABELS)


def render(ownership: str) -> str:
    return render_codeowners(load(ownership))


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
# owner: @alice @pytorch/dev-infra
# /torch/_dynamo/
/torch/_dynamo/eval_frame.py @alice someone@example.com
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
        ownership = replace_once(OWNERSHIP, '  "module: tvm": "module: dynamo"\n', "")
        text = render(replace_once(ownership, '["module: tvm"]', "[]"))
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
            "sits inside /torch/_dynamo/ in 'dynamo'": (
                "/torch/_dynamo_unlabeled.py",
                "/torch/_dynamo/unlabeled.py",
            ),
            "section_notes for sections not in `sections`": (
                "  protected_codeownership:\n    - ",
                "  protected:\n    - ",
            ),
            "not in tools/linter/adapters/pytorch_labels.json": (
                '"module: tvm": "module: dynamo"',
                '"module: tvm": "module: dynamoo"',
            ),
            "cycle through 'module: dynamo'": (
                '"module: dynamo": "oncall: pt2"',
                '"module: dynamo": "oncall: pt2"\n  "oncall: pt2": "module: dynamo"',
            ),
            "cycle through 'module: nn'": (
                '"module: linear cross entropy": "module: nn"',
                '"module: linear cross entropy": "module: nn"\n  "module: nn": "module: nn"',
            ),
            "'module: lce' is not in": (
                '"module: linear cross entropy": "module: nn"',
                '"module: lce": "module: nn"',
            ),
            "parent 'module: tvm' is not in": (
                '"module: linear cross entropy": "module: nn"',
                '"module: linear cross entropy": "module: tvm"',
            ),
            "expected 'label: parent'": (
                '"module: tvm": "module: dynamo"',
                '"module: tvm": ["module: dynamo"]',
            ),
        }
        for message, (old, new) in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    render(replace_once(OWNERSHIP, old, new))

    def test_label_rules(self):
        self.assertEqual(
            splice_label_rules(LABEL_RULES, load(OWNERSHIP)),
            f"""\
- any:
  - "good first issue"
  then:
  - "actionable"
{LABEL_RULES_BEGIN}
- any:
  - "module: dynamo"
  then:
  - "oncall: pt2"
- any:
  - "module: tvm"
  then:
  - "module: dynamo"
- any:
  - "module: linear cross entropy"
  then:
  - "module: nn"
{LABEL_RULES_END}
""",
        )
        reversed_markers = f"{LABEL_RULES_END}\n{LABEL_RULES_BEGIN}\n"
        for current in ("- any: []\n", reversed_markers, LABEL_RULES + LABEL_RULES):
            with self.subTest(current=current):
                with self.assertRaisesRegex(ValueError, "needs one line"):
                    splice_label_rules(current, load(OWNERSHIP))

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


def run_generator(
    *, ownership: str, codeowners: str, args: list[str], label_rules: str = LABEL_RULES
):
    """Run the generator in a throwaway repo; return the result and final CODEOWNERS and label rules."""
    with tempfile.TemporaryDirectory() as repo:
        Path(repo, "OWNERSHIP.yaml").write_text(ownership)
        Path(repo, "CODEOWNERS").write_text(codeowners)
        Path(repo, ".github").mkdir()
        Path(repo, ".github/label_to_label.yml").write_text(label_rules)
        Path(repo, "tools/linter/adapters").mkdir(parents=True)
        Path(repo, "tools/linter/adapters/pytorch_labels.json").write_text(
            json.dumps(sorted(KNOWN_LABELS))
        )
        subprocess.check_output(["git", "init", "-q"], cwd=repo)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        return (
            result,
            Path(repo, "CODEOWNERS").read_text(),
            Path(repo, ".github/label_to_label.yml").read_text(),
        )


class TestGeneratorModes(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_up_to_date(self):
        generated = render(OWNERSHIP)
        label_rules = splice_label_rules(LABEL_RULES, load(OWNERSHIP))
        for args in (["--check"], ["--lintrunner"]):
            result, _, _ = run_generator(
                ownership=OWNERSHIP,
                codeowners=generated,
                label_rules=label_rules,
                args=args,
            )
            self.assertEqual((result.returncode, result.stdout), (0, ""), result.stderr)

    def test_stale_outputs(self):
        generated = render(OWNERSHIP)
        label_rules = splice_label_rules(LABEL_RULES, load(OWNERSHIP))
        result, _, _ = run_generator(
            ownership=OWNERSHIP, codeowners="stale\n", args=["--check"]
        )
        self.assertEqual(result.returncode, 1)
        result, _, _ = run_generator(
            ownership=OWNERSHIP, codeowners="stale\n", args=["--lintrunner"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(
            [(m["path"], m["code"], m["original"], m["replacement"]) for m in messages],
            [
                ("CODEOWNERS", "CODEOWNERS_GENERATED", "stale\n", generated),
                (
                    ".github/label_to_label.yml",
                    "CODEOWNERS_GENERATED",
                    LABEL_RULES,
                    label_rules,
                ),
            ],
        )
        result, codeowners, written_rules = run_generator(
            ownership=OWNERSHIP, codeowners="stale\n", args=[]
        )
        self.assertEqual(
            (result.returncode, codeowners, written_rules), (0, generated, label_rules)
        )

    def test_invalid_ownership(self):
        invalid = replace_once(OWNERSHIP, "name: nn\n", "name: dynamo\n")
        result, _, _ = run_generator(
            ownership=invalid, codeowners="x\n", args=["--check"]
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("duplicate OWNERSHIP.yaml names", result.stderr)
        result, _, _ = run_generator(
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

# Owner(s): ["module: ci"]

import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/codeowners"))
from generate_codeowners import load_ownership, render_codeowners, SECTION_MARKER


OWNERSHIP = """\
oncall_queues:
  "oncall: pt2": ["module: dynamo"]
planned_labels: ["module: tvm"]
categories:
  - name: "Compiler"
    sections: [compiler_dynamo]
  - name: "Protected ownership"
    sections: [protected_codeownership]
sections:
  - name: compiler_dynamo
    paths:
      - /torch/_dynamo/unlabeled.py
    areas:
      - labels: ["module: dynamo"]
        owners: [alice, pytorch/dev-infra]
        paths:
          - /torch/_dynamo/
          - /torch/_dynamo/eval_frame.py: [alice, someone@example.com]
      - labels: ["module: dynamo", "module: tvm"]
        owners: []
        paths:
          - /torch/_dynamo/backends/tvm.py
  - name: protected_codeownership
    note:
      - "# Keep these rules last."
    paths:
      - /CODEOWNERS: [albanD]
"""


class TestGenerateCodeowners(TestCase):
    def test_render(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "OWNERSHIP.yaml")
            path.write_text(OWNERSHIP)
            text = render_codeowners(load_ownership(path))
        header, sections = text.split(SECTION_MARKER + "\n")
        self.assertIn("#     oncall: pt2: dynamo\n", header)
        self.assertIn("#   them until they exist): module: tvm\n", header)
        self.assertIn("# - Compiler: compiler_dynamo\n", header)
        self.assertEqual(
            sections,
            """\
#
# [compiler_dynamo]
# /torch/_dynamo/unlabeled.py
# label: module: dynamo
# owner: @alice @pytorch/dev-infra
# /torch/_dynamo/
/torch/_dynamo/eval_frame.py @alice someone@example.com
# label: module: dynamo, module: tvm
# owner: none
# /torch/_dynamo/backends/tvm.py
#
# [protected_codeownership]
# Keep these rules last.
/CODEOWNERS @albanD
#
""",
        )


if __name__ == "__main__":
    run_tests()

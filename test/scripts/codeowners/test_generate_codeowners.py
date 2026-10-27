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
sections:
  - compiler_dynamo
  - python_nn_api
  - protected_codeownership
section_notes:
  protected_codeownership:
    - "# Keep these rules last."
ownership:
  - labels: ["module: linear cross entropy"]
    owners: [pearu]
    section: python_nn_api
    paths:
      - /torch/nn/modules/linear_cross_entropy.py: [pearu]
  - labels: ["module: dynamo"]
    owners: [alice, pytorch/dev-infra]
    section: compiler_dynamo
    paths:
      - /torch/_dynamo/
      - /torch/_dynamo/eval_frame.py: [alice, someone@example.com]
      - /torch/nn/dynamo_hooks.py: {section: python_nn_api}
  - labels: ["module: dynamo", "module: tvm"]
    owners: []
    section: compiler_dynamo
    paths:
      - /torch/_dynamo/backends/tvm.py
  - labels: ["module: nn"]
    owners: [bob]
    section: python_nn_api
    paths:
      - /torch/nn/: [bob]
  - section: compiler_dynamo
    paths:
      - /torch/_dynamo/unlabeled.py
  - section: protected_codeownership
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
        self.assertIn(
            "# - Sections: compiler_dynamo, python_nn_api, protected_codeownership\n",
            header,
        )
        # Paths without a label group come first in their section, and a group's
        # path can override its section. In python_nn_api the nn block (/torch/nn/)
        # comes before the two blocks with paths inside it, even though
        # OWNERSHIP.yaml lists it last; those two keep their OWNERSHIP.yaml order.
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
# [python_nn_api]
# label: module: nn
# owner: @bob
/torch/nn/ @bob
# label: module: linear cross entropy
# owner: @pearu
/torch/nn/modules/linear_cross_entropy.py @pearu
# label: module: dynamo
# owner: @alice @pytorch/dev-infra
# /torch/nn/dynamo_hooks.py
#
# [protected_codeownership]
# Keep these rules last.
/CODEOWNERS @albanD
#
""",
        )


if __name__ == "__main__":
    run_tests()

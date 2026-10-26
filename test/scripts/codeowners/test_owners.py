# Owner(s): ["module: ci"]

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


class TestCollectTestOwners(TestCase):
    def test_header_labels_then_codeowners(self):
        codeowners = """\
# Grouped review-surface definitions:

# [infra_test]
# label: module: ci
/test/ @owner

# [compiler_dynamo]
# label: oncall: pt2, module: dynamo
/test/dynamo/ @owner

# [core_tests]
# label: module: unknown
/test/core/ @owner
"""
        files = {
            "CODEOWNERS": codeowners,
            "c10/test/test_unowned.cpp": "// Native tests carry no owner header.\n",
            "test/core/test_ignored_label.py": "import torch\n",
            "test/dynamo/test_empty_header.py": "# Owner(s): []\n",
            "test/dynamo/test_filtered_header.py": '# Owner(s): ["module: unknown", "module: dynamo", "module: ci"]\n',
            "test/dynamo/test_header_wins.py": '# Owner(s): ["module: inductor"]\n',
            "test/dynamo/test_ignored_header.py": '# Owner(s): ["oncall: pt2", "module: tests"]\n',
            "test/fb/test_excluded.py": '# Owner(s): ["module: ci"]\n',
            "test/helper.py": '# Owner(s): ["module: ci"]\n',
            "test/test_no_header.py": "import torch\n",
        }
        script = Path(__file__).resolve().parents[3] / "scripts/codeowners/owners.py"
        with tempfile.TemporaryDirectory() as repo:
            for path, content in files.items():
                file = Path(repo, path)
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(content)
            identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
            for command in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "init"]):
                subprocess.check_output(["git", *identity, *command], cwd=repo)
            command = [sys.executable, "-S", str(script)]
            output = subprocess.check_output(command, cwd=repo, text=True)
        self.assertEqual(
            json.loads(output),
            {
                "c10/test/test_unowned.cpp": [],
                "test/core/test_ignored_label.py": ["core_tests"],
                "test/dynamo/test_empty_header.py": ["dynamo"],
                "test/dynamo/test_filtered_header.py": ["ci", "dynamo"],
                "test/dynamo/test_header_wins.py": ["inductor"],
                "test/dynamo/test_ignored_header.py": ["dynamo"],
                "test/test_no_header.py": ["ci"],
            },
        )


if __name__ == "__main__":
    run_tests()

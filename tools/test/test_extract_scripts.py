from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.extract_scripts import main


class TestExtractScripts(unittest.TestCase):
    def test_index_prefix_padded_to_widest_step(self) -> None:
        # A job with exactly ten script steps sits on the power-of-ten
        # boundary: the indices run 1..10, so every filename prefix must be two
        # digits for the extracted files to sort in step order.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflows = root / ".github" / "workflows"
            workflows.mkdir(parents=True)
            steps = "\n".join(f"      - run: echo {n}" for n in range(10))
            (workflows / "ci.yml").write_text(f"jobs:\n  build:\n    steps:\n{steps}\n")
            out = root / "out"

            cwd = os.getcwd()
            os.chdir(root)
            try:
                with mock.patch.object(
                    sys, "argv", ["extract_scripts.py", "--out", str(out)]
                ):
                    main()
            finally:
                os.chdir(cwd)

            job_dir = out / ".github" / "workflows" / "ci.yml" / "build"
            names = sorted(p.name for p in job_dir.iterdir())
            self.assertEqual(len(names), 10)
            # The first and tenth steps must share the same prefix width, else
            # "10.sh" would sort between "1.sh" and "2.sh".
            self.assertIn("01.sh", names)
            self.assertIn("10.sh", names)
            self.assertTrue(all(re.fullmatch(r"\d{2}\.sh", n) for n in names), names)


if __name__ == "__main__":
    unittest.main()

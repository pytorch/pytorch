"""Fail closed if a search-tool download does not match its pinned digest."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


INSTALLER = Path(__file__).resolve().parents[1] / "install-tools.sh"


@unittest.skipUnless(sys.platform == "linux", "installer targets Linux runners")
class SearchToolInstallTest(unittest.TestCase):
    def test_bad_download_never_installs_or_publishes_a_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shims = root / "shims"
            shims.mkdir()
            curl = shims / "curl"
            curl.write_text('#!/bin/bash\nprintf corrupt > "${@: -1}"\n')
            curl.chmod(0o755)
            path_file = root / "github-path"
            path_file.touch()
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                env={
                    **os.environ,
                    "PATH": f"{shims}:{os.environ['PATH']}",
                    "RUNNER_TEMP": str(root),
                    "GITHUB_PATH": str(path_file),
                },
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 mismatch", result.stdout)
            self.assertFalse((root / "pi-tools/bin/rg").exists())
            self.assertFalse((root / "pi-tools/bin/fd").exists())
            self.assertEqual(path_file.read_text(), "")


if __name__ == "__main__":
    unittest.main()

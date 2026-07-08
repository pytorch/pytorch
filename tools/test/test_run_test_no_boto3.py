from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# run_test.py must stay usable when boto3 is absent (e.g. a contributor's
# machine): boto3 is only used to upload metrics/stats to S3 and its import is
# guarded in tools/stats/upload_metrics.py. run_test.py itself can't be imported
# without a built torch, but the only boto3 dependency in its import chain lives
# in tools/stats, so we exercise that here in a fresh interpreter with boto3
# blocked via sys.modules[...] = None (which makes `import boto3` raise).
_CHECK = """
import sys
sys.modules["boto3"] = None
from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_CLASS_TIMES_FILE,
    TEST_TIMES_FILE,
)
from tools.stats.upload_metrics import add_global_metric, emit_metric
import tools.stats.upload_metrics as m
assert m.EMIT_METRICS is False, f"EMIT_METRICS should be False without boto3, got {m.EMIT_METRICS}"
"""


class TestRunTestNoBoto3(unittest.TestCase):
    def test_stats_imports_without_boto3(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", _CHECK],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"run_test.py stats imports failed without boto3:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

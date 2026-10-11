from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats import upload_artifacts


sys.path.remove(str(REPO_ROOT))

REPO = "pytorch/pytorch"
RUN_ID = 37994614952
REPORTS = [
    "distributed.test_c10d_nccl-2222.report.jsonl",
    "test_torch-1111.report.jsonl",
]


class TestUploadTorchciReports(unittest.TestCase):
    def _upload(self, job_id: str) -> list[str]:
        name = f"torchci-reports-runattempt1-{job_id}.zip"
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / name
            with zipfile.ZipFile(artifact, "w") as z:
                for report in REPORTS:
                    z.writestr(report, "{}\n")
            with mock.patch.object(upload_artifacts, "upload_file_to_s3") as upload:
                upload_artifacts.upload_torchci_reports(REPO, RUN_ID, artifact)
        return sorted(call.kwargs["key"] for call in upload.call_args_list)

    def test_reports_get_the_keys_the_action_uses(self) -> None:
        folder = f"torchci-reports/{REPO}/{RUN_ID}/114041013692"
        expected = [f"{folder}/{report}" for report in REPORTS]
        self.assertEqual(self._upload("114041013692"), expected)

    def test_no_job_id_uploads_nothing(self) -> None:
        self.assertEqual(self._upload(""), [])


if __name__ == "__main__":
    unittest.main()

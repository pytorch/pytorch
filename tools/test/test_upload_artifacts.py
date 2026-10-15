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
JOB_ID = "114041013692"
REPORTS = [
    "distributed.test_c10d_nccl-2222.report.jsonl",
    "test_torch-1111.report.jsonl",
]


class TestUploadTorchciReports(unittest.TestCase):
    def _upload(
        self,
        job_id: str = JOB_ID,
        run_attempt: int = 1,
        download_failed: bool = False,
        s3_error: Exception | None = None,
    ) -> list[str]:
        name = f"torchci-reports-runattempt1-{job_id}.zip"
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / name
            if download_failed:
                # What _download_artifact saves when GitHub returns an error.
                artifact.write_text('{"message":"Artifact has expired"}')
            else:
                with zipfile.ZipFile(artifact, "w") as z:
                    for report in REPORTS:
                        z.writestr(report, "{}\n")
            with mock.patch.object(
                upload_artifacts, "upload_file_to_s3", side_effect=s3_error
            ) as upload:
                upload_artifacts.upload_torchci_reports(
                    REPO, RUN_ID, run_attempt, artifact
                )
        return sorted(call.kwargs["key"] for call in upload.call_args_list)

    def test_reports_get_the_keys_the_action_uses(self) -> None:
        folder = f"torchci-reports/{REPO}/{RUN_ID}/{JOB_ID}"
        expected = [f"{folder}/{report}" for report in REPORTS]
        self.assertEqual(self._upload(), expected)

    def test_no_job_id_uploads_nothing(self) -> None:
        self.assertEqual(self._upload(job_id=""), [])

    def test_other_attempts_upload_nothing(self) -> None:
        self.assertEqual(self._upload(run_attempt=2), [])

    def test_failed_download_uploads_nothing(self) -> None:
        self.assertEqual(self._upload(download_failed=True), [])

    def test_failed_upload_stops_without_raising(self) -> None:
        self.assertEqual(len(self._upload(s3_error=RuntimeError("S3"))), 1)


if __name__ == "__main__":
    unittest.main()

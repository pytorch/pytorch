import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.testing.upload_artifacts import upload_to_s3_artifacts


class TestUploadArtifacts(unittest.TestCase):
    def test_upload_to_s3_artifacts_uploads_test_jsons_zip(self) -> None:
        suffix = "linux-test"

        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            reports_dir = repo_root / "test" / "test-reports" / "suite"
            reports_dir.mkdir(parents=True)
            (reports_dir / "report.xml").write_text("<testsuite />")
            (reports_dir / "report.csv").write_text("name,status\n")
            (reports_dir / "report.log").write_text("test log\n")
            (reports_dir / "report.json").write_text("{}\n")

            mock_s3 = mock.Mock()
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "GITHUB_RUN_ID": "123",
                        "GITHUB_RUN_ATTEMPT": "2",
                        "ARTIFACTS_FILE_SUFFIX": suffix,
                    },
                    clear=True,
                ),
                mock.patch("tools.testing.upload_artifacts.REPO_ROOT", repo_root),
                mock.patch(
                    "tools.testing.upload_artifacts.get_s3_resource",
                    return_value=mock_s3,
                ),
            ):
                upload_to_s3_artifacts(failed=False)

            uploaded = mock_s3.upload_file.call_args_list
            self.assertEqual(len(uploaded), 3)

            upload_file, _, upload_key = uploaded[2].args
            self.assertEqual(Path(upload_file).name, f"test-jsons-{suffix}.zip")
            self.assertTrue(upload_key.endswith(f"/test-jsons-{suffix}.zip"))

            with zipfile.ZipFile(upload_file) as jsons_zip:
                self.assertEqual(
                    jsons_zip.namelist(), ["test/test-reports/suite/report.json"]
                )


if __name__ == "__main__":
    unittest.main()

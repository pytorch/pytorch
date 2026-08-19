import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.dynamo import verify_dynamo

import torch
from torch.torch_version import TorchVersion


class TestVerifyDynamoRocmVersion(unittest.TestCase):
    def test_reads_full_rocm_sdk_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            header = Path(tmpdir) / "include" / "rocm-core" / "rocm_version.h"
            header.parent.mkdir(parents=True)
            header.write_text(
                "#define ROCM_VERSION_MAJOR 10\n"
                "#define ROCM_VERSION_MINOR 1\n"
                "#define ROCM_VERSION_PATCH 2\n"
            )
            with mock.patch.object(
                verify_dynamo, "_find_rocm_home", return_value=(tmpdir, None)
            ):
                self.assertEqual(
                    verify_dynamo.get_rocm_sdk_version(), TorchVersion("10.1.2")
                )

    def test_malformed_rocm_sdk_header_falls_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            header = Path(tmpdir) / "include" / "rocm-core" / "rocm_version.h"
            header.parent.mkdir(parents=True)
            header.write_text(
                "#define ROCM_VERSION_MAJOR 10\n#define ROCM_VERSION_MINOR 1\n"
            )
            with mock.patch.object(
                verify_dynamo, "_find_rocm_home", return_value=(tmpdir, None)
            ):
                self.assertIsNone(verify_dynamo.get_rocm_sdk_version())

    def test_missing_rocm_sdk_header_falls_back(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(
                verify_dynamo, "_find_rocm_home", return_value=(tmpdir, None)
            ),
        ):
            self.assertIsNone(verify_dynamo.get_rocm_sdk_version())

    def test_check_rocm_compares_full_sdk_versions(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.version, "hip", "7.15.26306"),
            mock.patch.object(torch.version, "rocm", "10.1.0"),
            mock.patch.object(
                verify_dynamo,
                "get_rocm_sdk_version",
                return_value=TorchVersion("10.1.1"),
            ),
            self.assertWarnsRegex(UserWarning, "ROCm version mismatch"),
        ):
            self.assertEqual(verify_dynamo.check_rocm(), TorchVersion("10.1.1"))

    def test_check_rocm_falls_back_to_hip_for_old_wheel(self):
        old_rocm = torch.version.rocm
        delattr(torch.version, "rocm")
        try:
            with (
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(torch.version, "hip", "7.15.26306"),
                mock.patch.object(
                    verify_dynamo, "get_rocm_sdk_version", return_value=None
                ),
                mock.patch.object(
                    verify_dynamo,
                    "get_hip_version",
                    return_value=TorchVersion("7.15"),
                ) as get_hip_version,
            ):
                self.assertEqual(verify_dynamo.check_rocm(), TorchVersion("7.15"))
                get_hip_version.assert_called_once_with()
        finally:
            torch.version.rocm = old_rocm


if __name__ == "__main__":
    unittest.main()

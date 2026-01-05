# Owner(s): ["module: cuda"]

import warnings
from unittest.mock import patch

import torch
import torch.cuda
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCodeCompatibleWithDevice(TestCase):
    def test_compatible_cases(self):
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=80, code_cc=80)
        )
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=86, code_cc=80)
        )

    def test_backward_incompatible(self):
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=80, code_cc=86)
        )

    def test_cross_major_incompatible(self):
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=90, code_cc=80)
        )
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=75, code_cc=80)
        )

    def test_igpu_cases(self):
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=53, code_cc=50)
        )
        self.assertFalse(
            torch.cuda._code_compatible_with_device(device_cc=87, code_cc=80)
        )
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=53, code_cc=53)
        )

    def test_special_case_sm101_on_sm110(self):
        self.assertTrue(
            torch.cuda._code_compatible_with_device(device_cc=110, code_cc=101)
        )

    def test_unknown_code_cc(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cuda._code_compatible_with_device(device_cc=990, code_cc=990)
            self.assertTrue(result)
            self.assertEqual(len(w), 1)
            self.assertIn("unknown compute capability", str(w[0].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = torch.cuda._code_compatible_with_device(device_cc=991, code_cc=990)
            self.assertTrue(result)
            self.assertEqual(len(w), 1)


@patch("torch.cuda.get_device_name", return_value="NVIDIA MOCK DEVICE")
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.version.cuda", "12.6")
class TestCheckCapability(TestCase):
    def test_rocm_skips_check(self, *args):
        with (
            patch("torch.version.cuda", None),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            self.assertIsNone(torch.version.cuda)
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_70", "sm_80", "sm_90"])
    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_compatible_device_no_warning(self, *args):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_80"])
    @patch("torch.cuda.get_device_capability", return_value=(7, 5))
    def test_incompatible_device_warns(self, *args):
        with self.assertWarnsRegex(
            UserWarning, r"Found GPU0.*which is of compute capability.*7\.5"
        ):
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_80"])
    @patch("torch.cuda.get_device_capability", return_value=(8, 7))
    def test_incompatible_device_warns_igpu(self, *args):
        with self.assertWarnsRegex(
            UserWarning, r"Found GPU0.*which is of compute capability.*8\.7"
        ):
            torch.cuda._check_capability()

    @patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"])
    def test_multiple_devices_mixed_compatibility(self, *args):
        caps = [(8, 0), (7, 5), (8, 6)]
        with (
            patch("torch.cuda.device_count", return_value=len(caps)),
            patch("torch.cuda.get_device_capability", side_effect=caps),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            self.assertIn("GPU1", str(w[0].message))

    @patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"])
    @patch("torch.cuda.get_device_capability", return_value=(7, 5))
    def test_warning_message_contains_device_info(self, *args):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
            self.assertIn("GPU0", msg)
            self.assertIn("NVIDIA MOCK DEVICE", msg)
            self.assertIn("compute capability (CC) 7.5", msg)
            self.assertIn("8.0 which supports", msg)
            self.assertIn("9.0 which supports", msg)

    @patch("torch.cuda.get_arch_list", return_value=["sm_60"])
    @patch("torch.cuda.get_device_capability", return_value=(7, 0))
    @patch(
        "torch.cuda.PYTORCH_RELEASES_CODE_CC",
        {"12.6": {50, 60, 70}, "12.8": {70}, "13.0": {75}},
    )
    def test_warning_suggests_compatible_pytorch_release(self, *args):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
            self.assertIn("12.6", msg)
            self.assertIn("12.8", msg)
            self.assertNotIn("13.0", msg)

    @patch("torch.cuda.get_arch_list", return_value=["sm_80"])
    @patch("torch.cuda.get_device_capability", return_value=(5, 3))
    def test_warning_no_compatible_pytorch_release(self, *args):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.cuda._check_capability()
            self.assertEqual(len(w), 1)
            msg = str(w[0].message)
            self.assertNotIn(
                "install a PyTorch release that supports one of these CUDA versions",
                msg,
            )


if __name__ == "__main__":
    run_tests()

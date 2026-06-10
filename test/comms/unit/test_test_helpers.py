#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys


# Make test/comms importable so `helpers` / `integration` resolve when this
# file is run directly (run_test.py runs `python comms/unit/<file>.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.comm_test_helpers import skip_if_torch_compile_not_supported_or_enabled

from torch.testing._internal.common_utils import run_tests, TestCase


class TestSkipIfTorchCompileNotSupportedOrEnabled(TestCase):
    def setUp(self):
        # Save original env vars
        self._orig_patch = os.environ.get("TORCHCOMMS_PATCH_FOR_COMPILE")
        self._orig_ignore = os.environ.get(
            "TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT"
        )
        # Enable compile mode for tests
        os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"
        # Clear ignore flag
        if "TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT" in os.environ:
            del os.environ["TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT"]

    def tearDown(self):
        # Restore original env vars
        if self._orig_patch is not None:
            os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = self._orig_patch
        elif "TORCHCOMMS_PATCH_FOR_COMPILE" in os.environ:
            del os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"]
        if self._orig_ignore is not None:
            os.environ["TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT"] = (
                self._orig_ignore
            )
        elif "TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT" in os.environ:
            del os.environ["TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT"]

    def test_passes_when_version_meets_requirement(self):
        """Test that decorated function runs normally when version is sufficient."""

        @skip_if_torch_compile_not_supported_or_enabled(_current_version="2.12.0")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")
        self.assertFalse(getattr(dummy_test, "__unittest_skip__", False))

    def test_skips_when_version_below_requirement(self):
        """Test that decorated function is skipped when version is too low."""

        @skip_if_torch_compile_not_supported_or_enabled(_current_version="2.10.0")
        def dummy_test(self):
            pass

        self.assertTrue(getattr(dummy_test, "__unittest_skip__", False))

    def test_skips_when_compile_not_enabled(self):
        """Test that decorated function is skipped when TORCHCOMMS_PATCH_FOR_COMPILE is not set."""
        del os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"]

        @skip_if_torch_compile_not_supported_or_enabled(_current_version="2.12.0")
        def dummy_test(self):
            pass

        self.assertTrue(getattr(dummy_test, "__unittest_skip__", False))

    def test_handles_version_with_cuda_suffix(self):
        """Test that version parsing handles CUDA suffixes like 2.12.0+cu118."""

        @skip_if_torch_compile_not_supported_or_enabled(_current_version="2.12.0+cu118")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")

    def test_handles_dev_version(self):
        """Test that version parsing handles dev versions like 2.12.0.dev20240101."""

        @skip_if_torch_compile_not_supported_or_enabled(
            _current_version="2.12.0.dev20240101"
        )
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")

    def test_compares_minor_version_correctly(self):
        """Test that 2.11 < 2.12."""

        @skip_if_torch_compile_not_supported_or_enabled(_current_version="2.11.0")
        def dummy_test(self):
            pass

        self.assertTrue(getattr(dummy_test, "__unittest_skip__", False))

    def test_compares_major_version_correctly(self):
        """Test that 3.0 > 2.12."""

        @skip_if_torch_compile_not_supported_or_enabled(_current_version="3.0.0")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")

    def test_ignore_version_requirement_env_var(self):
        """Test that TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT=1 bypasses version check."""
        os.environ["TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT"] = "1"

        @skip_if_torch_compile_not_supported_or_enabled(_current_version="2.10.0")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")


if __name__ == "__main__":
    run_tests()

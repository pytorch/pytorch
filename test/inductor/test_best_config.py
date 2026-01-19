# Owner(s): ["module: inductor"]

import glob
import json
import os
import sys
import tempfile
import unittest

import torch
from torch._inductor import config
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


try:
    import triton  # noqa: F401
except ImportError as e:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton") from e

from torch._inductor.test_case import run_tests, TestCase


def trivial_kernel(x):
    return torch.sin(x) + torch.cos(x)


class TestKernelBestConfig(TestCase):
    device_type = GPU_TYPE

    @classmethod
    def setUpClass(cls):
        # Save the original configuration and environment variables.
        cls.original_compile_threads = config.compile_threads
        cls.original_max_autotune = config.max_autotune
        cls.original_inductor_env = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
        cls.original_triton_env = os.environ.get("TRITON_CACHE_DIR", "")
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        # Restore the original configuration and environment variables.
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cls.original_inductor_env
        os.environ["TRITON_CACHE_DIR"] = cls.original_triton_env
        config.compile_threads = cls.original_compile_threads
        config.max_autotune = cls.original_max_autotune
        super().tearDownClass()

    def test_best_config_has_triton_cache_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = tmpdir
            triton_cache_dir = os.path.join(tmpdir, "triton_cache")
            os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

            config.compile_threads = 0
            config.max_autotune = True

            compiled_fn = torch.compile(trivial_kernel)

            x = torch.randn(32, 10, device=GPU_TYPE)
            compiled_fn(x)

            # Search for .best_config files in the inductor cache directory.
            best_config_files = glob.glob(
                os.path.join(tmpdir, "**", "*.best_config"), recursive=True
            )
            self.assertGreater(
                len(best_config_files),
                0,
                f"No best_config files found in {tmpdir}. Directory contents: {os.listdir(tmpdir)}",
            )

            # Validate that each best_config file contains a real triton_cache_hash,
            # and that a corresponding Triton cache directory exists.
            for file_path in best_config_files:
                with open(file_path) as f:
                    data = json.load(f)
                self.assertIn(
                    "triton_cache_hash",
                    data,
                    f"Missing triton_cache_hash in {os.path.basename(file_path)}",
                )
                cache_hash = data["triton_cache_hash"]
                expected_path = os.path.join(triton_cache_dir, cache_hash)
                self.assertTrue(
                    os.path.exists(expected_path),
                    f"Triton cache directory missing: {expected_path}",
                )


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()

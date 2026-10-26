from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

from tools.ramjet_run import distributed_environment, main

from torch.testing._internal import tci
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestRamjetRun(TestCase):
    root: Path

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-run-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @parametrize(
        "file",
        [
            "test/distributed/test_distributed_spawn.py",
            "test/distributed/algorithms/quantization/test_quantization.py",
        ],
    )
    @parametrize("collect", [False, True])
    def test_cpu_harness_wraps_collection_and_execution(self, file, collect):
        directory = None

        def run_path(path, *, run_name):
            nonlocal directory
            self.assertEqual(path, str(self.root / file))
            self.assertEqual(run_name, "__main__")
            self.assertEqual(os.environ["BACKEND"], "gloo")
            self.assertEqual(os.environ["WORLD_SIZE"], "3")
            self.assertEqual(
                os.environ["TEST_REPORT_SOURCE_OVERRIDE"], "dist-gloo-init-file"
            )
            directory = Path(os.environ["TEMP_DIR"])
            self.assertTrue(directory.joinpath("barrier").is_dir())
            self.assertTrue(directory.joinpath("test_dir").is_dir())

        arguments = ["launcher", str(self.root / file), "--use-pytest"]
        if collect:
            arguments.append("--list-markers")
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(sys, "argv", arguments),
            mock.patch.object(sys, "path", list(sys.path)),
            mock.patch.object(
                tci, "current_platform", return_value=tci.platform.linux_x86_64
            ),
            mock.patch("torch.cuda.device_count") as count,
            mock.patch("runpy.run_path", side_effect=run_path),
        ):
            main()
            self.assertNotIn("BACKEND", os.environ)
            self.assertNotIn("WORLD_SIZE", os.environ)
            self.assertNotIn("TEMP_DIR", os.environ)
            self.assertNotIn("TEST_REPORT_SOURCE_OVERRIDE", os.environ)
        count.assert_not_called()
        self.assertIsNotNone(directory)
        self.assertFalse(directory.exists())

    @parametrize("devices", [1, 4])
    def test_gpu_harness_uses_nccl(self, devices):
        script = self.root / "test/distributed/test_distributed_spawn.py"
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                tci,
                "current_platform",
                return_value=tci.platform.linux_x86_64_cuda_l4,
            ),
            mock.patch("torch.cuda.device_count", return_value=devices),
            distributed_environment(script),
        ):
            self.assertEqual(os.environ["BACKEND"], "nccl")
            self.assertEqual(os.environ["WORLD_SIZE"], str(devices))
            self.assertEqual(
                os.environ["TEST_REPORT_SOURCE_OVERRIDE"], "dist-nccl-init-file"
            )

    def test_explicit_environment_survives_failure(self):
        script = self.root / "test/distributed/test_distributed_spawn.py"
        environment = {
            "BACKEND": "gloo",
            "WORLD_SIZE": "2",
            "TEMP_DIR": str(self.root / "explicit"),
            "TEST_REPORT_SOURCE_OVERRIDE": "custom-report",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(
                tci,
                "current_platform",
                return_value=tci.platform.linux_x86_64_cuda_l4,
            ),
            mock.patch("torch.cuda.device_count", return_value=1),
        ):
            with self.assertRaisesRegex(RuntimeError, "fixture failed"):
                with distributed_environment(script):
                    self.assertEqual(dict(os.environ), environment)
                    os.environ["WORLD_SIZE"] = "changed by test"
                    raise RuntimeError("fixture failed")
            self.assertEqual(dict(os.environ), environment)
            self.assertTrue(Path(environment["TEMP_DIR"]).is_dir())

    def test_other_tests_keep_environment(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(tci, "current_platform") as current,
            distributed_environment(self.root / "test/test_function_schema.py"),
        ):
            self.assertEqual(dict(os.environ), {})
        current.assert_not_called()


if __name__ == "__main__":
    run_tests()

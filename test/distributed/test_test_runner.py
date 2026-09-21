# Owner(s): ["oncall: distributed"]

import os
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test import run_test as runner

from torch.testing._internal.common_utils import run_tests, TestCase


class TestDistributedRunner(TestCase):
    def test_backend_runs_once(self):
        for module in (
            "distributed/test_distributed_spawn",
            "distributed/algorithms/quantization/test_quantization",
        ):
            for platform, expected in (
                ("linux", ["nccl", "gloo"]),
                ("win32", ["gloo"]),
            ):
                with self.subTest(module=module, platform=platform):
                    calls = []
                    directories = []
                    original = dict(os.environ)

                    def run_test(test_module, test_directory, options, **kwargs):
                        self.assertEqual(test_module, module)
                        self.assertEqual(
                            kwargs, {"extra_unittest_args": ["--subprocess"]}
                        )
                        backend = os.environ["BACKEND"]
                        calls.append(backend)
                        directory = os.environ["TEMP_DIR"]
                        directories.append(directory)
                        self.assertTrue(
                            os.path.isdir(os.path.join(directory, "barrier"))
                        )
                        self.assertTrue(
                            os.path.isdir(os.path.join(directory, "test_dir"))
                        )
                        self.assertEqual(os.environ["WORLD_SIZE"], "3")
                        self.assertEqual(
                            os.environ["TEST_REPORT_SOURCE_OVERRIDE"],
                            f"dist-{backend}-init-file",
                        )
                        self.assertEqual(
                            os.environ.get("INIT_METHOD"), original.get("INIT_METHOD")
                        )
                        return 0

                    with (
                        patch.object(runner, "sys", Namespace(platform=platform)),
                        patch.object(
                            runner,
                            "DISTRIBUTED_TESTS_CONFIG",
                            {
                                backend: {"WORLD_SIZE": "3"}
                                for backend in ("nccl", "gloo")
                            },
                        ),
                        patch.object(runner, "run_test", side_effect=run_test),
                    ):
                        self.assertEqual(
                            runner.test_distributed(
                                module, "test", Namespace(verbose=False)
                            ),
                            0,
                        )
                    self.assertEqual(calls, expected)
                    self.assertEqual(dict(os.environ), original)
                    self.assertTrue(
                        all(not os.path.exists(path) for path in directories)
                    )

    def test_failure_restores_environment(self):
        original = dict(os.environ)
        directories = []

        def fail(*args, **kwargs):
            directories.append(os.environ["TEMP_DIR"])
            os.environ["DISTRIBUTED_RUNNER_TEST"] = "modified"
            return 7

        with (
            patch.object(runner, "sys", Namespace(platform="linux")),
            patch.object(
                runner,
                "DISTRIBUTED_TESTS_CONFIG",
                {"gloo": {"WORLD_SIZE": "3"}, "nccl": {"WORLD_SIZE": "3"}},
            ),
            patch.object(runner, "run_test", side_effect=fail),
        ):
            self.assertEqual(
                runner.test_distributed(
                    "distributed/test_distributed_spawn",
                    "test",
                    Namespace(verbose=False),
                ),
                7,
            )
        self.assertEqual(len(directories), 1)
        self.assertFalse(os.path.exists(directories[0]))
        self.assertEqual(dict(os.environ), original)


if __name__ == "__main__":
    run_tests()

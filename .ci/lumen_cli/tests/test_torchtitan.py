import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

git_stub = types.ModuleType("git")
git_stub.GitCommandError = Exception
git_stub.RemoteProgress = object
git_stub.Repo = object
sys.modules.setdefault("git", git_stub)

import cli.lib.core.torchtitan.torchtitan_test as torchtitan_test


class TestTorchtitanTestRunner(unittest.TestCase):
    @patch.object(torchtitan_test, "working_directory", return_value=contextlib.nullcontext())
    @patch.object(torchtitan_test, "pip_install_packages")
    @patch.object(torchtitan_test, "_nightly_index_url", return_value="https://example.invalid")
    @patch.object(torchtitan_test, "clone_torchtitan")
    def test_prepare_installs_numpy_2_for_datasets_pyarrow(
        self, _clone, _index_url, mock_pip_install_packages, _working_directory
    ):
        runner = torchtitan_test.TorchtitanTestRunner(
            SimpleNamespace(test_plan="torchtitan_features_integration")
        )

        runner.prepare()

        self.assertEqual(
            mock_pip_install_packages.call_args_list,
            [
                call(
                    packages=[
                        "--pre",
                        "torchao",
                        "torchcomms",
                        "--index-url",
                        "https://example.invalid",
                    ],
                ),
                call(packages=["helion"]),
                call(packages=["numpy==2.0.2"]),
                call(packages=["-e", "."]),
                call(packages=["pytest", "pytest-cov"]),
            ],
        )

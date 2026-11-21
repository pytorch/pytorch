# Owner(s): ["module: dynamo"]

import csv
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import pandas as pd

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.dynamo import check_accuracy, common


sys.path.remove(str(REPO_ROOT))


requires_distributed = unittest.skipIf(
    not torch.distributed.is_available(), "requires distributed"
)


class BenchmarkRunnerTests(TestCase):
    def _benchmark_args(self, **overrides):
        values = {
            "accuracy": True,
            "ci": False,
            "devices": ["cpu"],
            "diff_branch": common.diff_branch_default,
            "performance": False,
            "timeout": 10,
        }
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_worker_failure_is_recorded(self):
        args = self._benchmark_args()
        runner = types.SimpleNamespace(suite_name="test")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "accuracy.csv"
            common.output_csv(
                output,
                ["dev", "name", "batch_size", "accuracy"],
                ["cpu", "model", 1, "pass"],
            )
            with (
                mock.patch.object(common, "output_filename", str(output)),
                mock.patch.object(common, "current_name", "model"),
                mock.patch.object(common, "output_signpost"),
                mock.patch.object(
                    common.subprocess,
                    "check_call",
                    side_effect=subprocess.CalledProcessError(1, "benchmark"),
                ),
            ):
                failures = common._run_model_in_subprocess(args, runner, "model")

            with output.open() as fd:
                rows = list(csv.DictReader(fd))

        self.assertEqual(failures, [("model", "cpu", "worker_fail")])
        self.assertEqual(rows[-1]["accuracy"], "worker_fail")

    def test_accuracy_uses_latest_result_and_reports_outcomes(self):
        actual = pd.DataFrame(
            [
                {"dev": "cuda", "name": "passed", "accuracy": "pass"},
                {"dev": "cuda", "name": "skipped", "accuracy": "pass_due_to_skip"},
                {"dev": "cuda", "name": "expected", "accuracy": "fail_to_run"},
                {"dev": "cuda", "name": "retried", "accuracy": "worker_fail"},
                {"dev": "cuda", "name": "retried", "accuracy": "pass"},
            ]
        )
        expected = pd.DataFrame(
            [
                {"name": "passed", "accuracy": "pass"},
                {"name": "skipped", "accuracy": "pass_due_to_skip"},
                {"name": "expected", "accuracy": "fail_to_run"},
                {"name": "retried", "accuracy": "pass"},
            ]
        )
        output = StringIO()

        with redirect_stdout(output):
            failed, _ = check_accuracy.check_accuracy(actual, expected, "expected.csv")

        self.assertFalse(failed)
        self.assertRegex(output.getvalue(), r"(?m)^passed\s+PASS$")
        self.assertRegex(output.getvalue(), r"(?m)^skipped\s+XFAIL$")
        self.assertRegex(output.getvalue(), r"(?m)^expected\s+XFAIL$")
        self.assertRegex(output.getvalue(), r"(?m)^retried\s+PASS$")

    def test_accuracy_rejects_expected_infrastructure_failure(self):
        actual = pd.DataFrame([{"name": "model", "accuracy": "worker_fail"}])
        expected = pd.DataFrame([{"name": "model", "accuracy": "worker_fail"}])

        failed, _ = check_accuracy.check_accuracy(actual, expected, "expected.csv")

        self.assertEqual(failed, ["model"])

    @requires_distributed
    def test_default_fsdp_policy_does_not_import_model_dependencies(self):
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        model_deps = frozenset({"diffusers", "torchbenchmark", "transformers"})
        with mock.patch.dict(sys.modules):
            for name in list(sys.modules):
                if name.partition(".")[0] in model_deps:
                    del sys.modules[name]
            policy = common.BenchmarkRunner().get_fsdp_auto_wrap_policy("resnet50")
            loaded = {name.partition(".")[0] for name in sys.modules} & model_deps

        self.assertEqual(loaded, set())
        self.assertIs(policy.func, size_based_auto_wrap_policy)
        self.assertEqual(policy.keywords["min_num_params"], int(1e5))

    @requires_distributed
    def test_diffusion_fsdp_policy_imports_current_model_class(self):
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        class Transformer2DModel(torch.nn.Module):
            pass

        module = types.SimpleNamespace(Transformer2DModel=Transformer2DModel)
        with mock.patch("importlib.import_module", return_value=module) as imported:
            policy = common.BenchmarkRunner().get_fsdp_auto_wrap_policy(
                "stable_diffusion_unet"
            )

        imported.assert_called_once_with("diffusers.models.transformers.transformer_2d")
        self.assertIsInstance(policy, ModuleWrapPolicy)
        self.assertTrue(policy(Transformer2DModel(), recurse=False))


if __name__ == "__main__":
    run_tests()

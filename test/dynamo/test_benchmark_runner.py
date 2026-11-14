# Owner(s): ["module: dynamo"]

import contextlib
import io
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

import pandas as pd

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.dynamo import check_graph_breaks, common
from benchmarks.dynamo.ci_expected_accuracy import update_expected


sys.path.remove(str(REPO_ROOT))


requires_distributed = unittest.skipIf(
    not torch.distributed.is_available(), "requires distributed"
)


class BenchmarkRunnerTests(TestCase):
    def test_get_dynamo_stats_recompiles(self):
        with mock.patch.dict(
            torch._dynamo.utils.counters, {"stats": Counter(recompiles=3)}
        ):
            stats = common.get_dynamo_stats()
        self.assertEqual(stats["recompiles"], 3)
        self.assertIsInstance(stats["recompiles"], int)

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


@instantiate_parametrized_tests
class RecompileMetricsTests(TestCase):
    @parametrize(
        "expected,actual,status",
        [
            (2, 2, "PASS"),
            (2, 3, "FAIL"),
            (2, 1, "IMPROVED"),
            (None, 3, "PASS"),
            (2, None, "PASS"),
            (float("nan"), 3, "PASS"),
            (2, float("nan"), "PASS"),
        ],
    )
    def test_compare_recompiles(self, expected, actual, status):
        baseline = pd.DataFrame([{"name": "model", "graph_breaks": 0}])
        result = pd.DataFrame(
            [{"name": "model", "graph_breaks": 0, "unique_graphs": 1}]
        )
        if expected is not None:
            baseline["recompiles"] = expected
        if actual is not None:
            result["recompiles"] = actual
        with contextlib.redirect_stdout(io.StringIO()) as output:
            changed, message = check_graph_breaks.check_graph_breaks(
                result, baseline, "expected.csv"
            )
        self.assertIn(status, output.getvalue())
        self.assertEqual(changed, [] if status == "PASS" else ["model"])
        if status == "FAIL":
            self.assertIn("Error:", message)
        elif status == "IMPROVED":
            self.assertIn("Improvement:", message)

    def test_compare_mixed_metrics(self):
        baseline = pd.DataFrame([{"name": "model", "graph_breaks": 2, "recompiles": 2}])
        result = pd.DataFrame(
            [{"name": "model", "graph_breaks": 1, "recompiles": 3, "unique_graphs": 1}]
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            changed, message = check_graph_breaks.check_graph_breaks(
                result, baseline, "expected.csv"
            )
        self.assertEqual(changed, ["model"])
        self.assertIn("IMPROVED:", output.getvalue())
        self.assertIn("FAIL:", output.getvalue())
        self.assertIn("Error:", message)
        self.assertNotIn("Improvement:", message)

    @parametrize("flaky,eager_failed", [(True, False), (False, True)])
    def test_compare_recompiles_skipped(self, flaky, eager_failed):
        baseline = pd.DataFrame([{"name": "model", "graph_breaks": 0, "recompiles": 2}])
        result = pd.DataFrame(
            [
                {
                    "name": "model",
                    "graph_breaks": 0,
                    "recompiles": 3,
                    "unique_graphs": 0 if eager_failed else 1,
                }
            ]
        )
        with (
            mock.patch.object(
                check_graph_breaks, "flaky_models", {"model"} if flaky else set()
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            changed, message = check_graph_breaks.check_graph_breaks(
                result, baseline, "expected.csv"
            )
        self.assertEqual(changed, [])
        self.assertEqual(message, "")
        self.assertIn("FAIL_BUT_FLAKY" if flaky else "EAGER_FAILED", output.getvalue())

    def test_download_preserves_unmeasured_recompiles(self):
        artifact = io.BytesIO()
        with ZipFile(artifact, "w") as archive:
            for phase in ("training", "inference"):
                archive.writestr(
                    f"test-reports/{phase}_torchbench.csv",
                    "name,accuracy,graph_breaks,recompiles\n"
                    "measured,pass,0,2\nunmeasured,pass,0,\nzero,pass,0,0\n",
                )
        artifact.seek(0)
        with mock.patch.object(update_expected, "urlopen", return_value=artifact):
            _, _, results = update_expected.download_single_artifact(
                "inductor_torchbench", 0, ["https://example.com/artifact.zip"]
            )
        result = results[("inductor_torchbench", "inference")].set_index("name")
        self.assertEqual(result.loc["measured", "recompiles"], 2)
        self.assertEqual(result.loc["zero", "recompiles"], 0)
        self.assertTrue(pd.isna(result.loc["unmeasured", "recompiles"]))

    def test_baseline_preserves_unmeasured_recompiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inductor_torchbench_inference.csv"
            path.write_text("name,accuracy,graph_breaks\nunmeasured,pass,0\n")
            result = pd.DataFrame(
                {
                    "name": ["measured", "zero"],
                    "accuracy": ["pass", "pass"],
                    "graph_breaks": [0, 0],
                    "recompiles": [2, 0],
                }
            )
            with mock.patch.object(update_expected, "apply_lints"):
                update_expected.write_filtered_csvs(
                    directory, {("inductor_torchbench", "inference"): result}
                )
            self.assertEqual(
                path.read_text(),
                "name,accuracy,graph_breaks,recompiles\n"
                "measured,pass,0,2\nunmeasured,pass,0,\nzero,pass,0,0\n",
            )


if __name__ == "__main__":
    run_tests()

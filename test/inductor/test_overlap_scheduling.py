# Owner(s): ["module: inductor"]

import operator
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.fx as fx
from torch._inductor.analysis import device_info
from torch._inductor.fx_passes.overlap_scheduling import gather_node_runtime_estimations
from torch._inductor.utils import get_device_tflops
from torch.testing._internal.common_utils import run_tests, TestCase


class TestOverlapSchedulingRuntimeEstimation(TestCase):
    def setUp(self):
        super().setUp()
        if hasattr(get_device_tflops, "cache_clear"):
            get_device_tflops.cache_clear()

    def tearDown(self):
        if hasattr(get_device_tflops, "cache_clear"):
            get_device_tflops.cache_clear()
        super().tearDown()

    def test_get_device_tflops_forwards_device_to_datasheet_tops(self):
        device = torch.device("cpu")

        with patch("torch._inductor.utils.datasheet_tops") as mock_datasheet_tops:
            mock_datasheet_tops.return_value = 123.0

            result = get_device_tflops(torch.float16, device=device)

        self.assertEqual(result, 123.0)
        mock_datasheet_tops.assert_called_once()
        self.assertEqual(mock_datasheet_tops.call_args.kwargs["device"], device)

    def test_get_device_tflops_non_cuda_without_datasheet_returns_zero(self):
        device = torch.device("cpu")

        with (
            patch("torch._inductor.utils.datasheet_tops", return_value=None),
            patch("torch.cuda.is_available", return_value=True),
        ):
            result = get_device_tflops(torch.float16, device=device)

        self.assertEqual(result, 0.0)

    def test_datasheet_tops_uses_device_name_helper(self):
        device = torch.device("cpu")
        fake_device_info = SimpleNamespace(
            tops={torch.float16: 456.0},
            tops_sparsity_factor=1.0,
        )

        with (
            patch.object(
                device_info,
                "_get_device_name",
                return_value="Fake Device",
            ) as mock_get_device_name,
            patch.object(
                device_info,
                "lookup_device_info",
                return_value=fake_device_info,
            ),
        ):
            result = device_info.datasheet_tops(
                torch.float16,
                device=device,
            )

        self.assertEqual(result, 456.0)
        mock_get_device_name.assert_called_once_with(device)

    def test_datasheet_tops_uses_tf32_key_for_tf32(self):
        device = torch.device("cpu")
        fake_device_info = SimpleNamespace(
            tops={"torch.tf32": 789.0},
            tops_sparsity_factor=1.0,
        )

        with (
            patch.object(device_info, "_get_device_name", return_value="Fake Device"),
            patch.object(
                device_info,
                "lookup_device_info",
                return_value=fake_device_info,
            ),
        ):
            result = device_info.datasheet_tops(
                torch.float32,
                is_tf32=True,
                device=device,
            )

        self.assertEqual(result, 789.0)

    def test_flops_to_ns_forwards_device_to_get_device_tflops(self):
        from torch.utils._runtime_estimation import flops_to_ns

        device = torch.device("cpu")

        with patch(
            "torch.utils._runtime_estimation.get_device_tflops",
            return_value=0.0,
        ) as mock_get_device_tflops:
            result = flops_to_ns(
                1000,
                torch.float16,
                device=device,
            )

        self.assertEqual(result, 0.0)
        mock_get_device_tflops.assert_called_once()
        self.assertEqual(mock_get_device_tflops.call_args.kwargs["device"], device)

    def test_gather_node_runtime_estimations_prefers_custom_estimation(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        relu = graph.call_function(torch.relu, (x,))
        graph.output(relu)
        gm = fx.GraphModule({}, graph)

        relu.meta["val"] = torch.empty(4, device="meta")

        def custom_runtime_estimation(
            node: fx.Node,
            size: int | None,
        ) -> float | None:
            if node is relu:
                return 7.0
            return None

        estimations = gather_node_runtime_estimations(
            gm,
            custom_runtime_estimation=custom_runtime_estimation,
            log_estimations=False,
        )

        self.assertEqual(estimations[relu], 7.0)

    def test_custom_estimator_skips_non_compute_roofline(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        getitem = graph.call_function(operator.getitem, ((x,), 0))
        graph.output(getitem)
        gm = fx.GraphModule({}, graph)

        getitem.meta["val"] = torch.empty(4, device="meta")

        def custom_runtime_estimation(
            node: fx.Node,
            size: int | None,
        ) -> float | None:
            return None

        with patch(
            "torch._inductor.fx_passes.overlap_scheduling.estimate_roofline_runtime_ms"
        ) as mock_estimate_roofline:
            estimations = gather_node_runtime_estimations(
                gm,
                custom_runtime_estimation=custom_runtime_estimation,
                log_estimations=False,
            )

        self.assertNotIn(getitem, estimations)
        mock_estimate_roofline.assert_not_called()

    def test_manual_overlap_scheduler_uses_noop_default_estimator(self):
        from torch._inductor.fx_passes import overlap_manual_scheduling

        graph = fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = fx.GraphModule({}, graph)

        captured_kwargs = {}

        def fake_overlap_scheduler_init(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            self.graph = gm.graph
            self.collective_info = {}

        with (
            patch.object(
                overlap_manual_scheduling.OverlapScheduler,
                "__init__",
                fake_overlap_scheduler_init,
            ),
            patch.object(
                overlap_manual_scheduling,
                "ManualOverlapPreservingBucketer",
            ),
        ):
            overlap_manual_scheduling.ManualOverlapScheduler(
                gm,
                module_bucket_plans=[],
                insert_overlap_deps=False,
            )

        estimator = captured_kwargs["custom_runtime_estimation"]
        self.assertIsNotNone(estimator)
        self.assertEqual(estimator(x, None), 0.0)


if __name__ == "__main__":
    run_tests()

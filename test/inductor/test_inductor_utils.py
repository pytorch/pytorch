# Owner(s): ["module: inductor"]

import functools
import logging
from unittest import mock

import torch
from torch._inductor import config, utils
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import do_bench_using_profiling
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

device_type = (
    acc.type
    if (acc := torch.accelerator.current_accelerator(check_available=True))
    else "cpu"
)


class TestFallbackByDefault(TestCase):
    def test_sym_size_uses_inductor_lowering_in_lite_mode(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        sym_size = graph.call_function(torch.ops.aten.sym_size.int, (x, 0))

        with config.patch({"fallback_by_default": True}):
            self.assertFalse(utils.should_fallback_by_default(sym_size))


class FakeKinetoEvent:
    def __init__(
        self,
        name: str,
        device_type: DeviceType,
        start_ns: int,
        end_ns: int,
        linked_correlation_id: int = 0,
        correlation_id: int = 0,
        activity_type: str = "kernel",
    ) -> None:
        self._name = name
        self._device_type = device_type
        self._start_ns = start_ns
        self._end_ns = end_ns
        self._linked_correlation_id = linked_correlation_id
        self._correlation_id = correlation_id
        self._activity_type = activity_type

    def name(self) -> str:
        return self._name

    def device_type(self) -> DeviceType:
        return self._device_type

    def start_ns(self) -> int:
        return self._start_ns

    def end_ns(self) -> int:
        return self._end_ns

    def linked_correlation_id(self) -> int:
        return self._linked_correlation_id

    def correlation_id(self) -> int:
        return self._correlation_id

    def activity_type(self) -> str:
        return self._activity_type


class FakeProfilerEvent:
    def __init__(
        self,
        name: str,
        device_type: DeviceType,
        id: int,
        cpu_children: list["FakeProfilerEvent"] | None = None,
    ) -> None:
        self.name = name
        self.device_type = device_type
        self.id = id
        self.cpu_children = cpu_children or []


class TestBench(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        x = torch.rand(1024, 10).to(device_type).half()
        w = torch.rand(512, 10).to(device_type).half()
        cls._bench_fn = staticmethod(
            functools.partial(torch.nn.functional.linear, x, w)
        )

    def test_benchmarker(self):
        res = benchmarker.benchmark_gpu(self._bench_fn)
        log.warning("do_bench result: %s", res)
        self.assertGreater(res, 0)

    def test_do_bench_using_profiling(self):
        res = do_bench_using_profiling(self._bench_fn)
        log.warning("do_bench_using_profiling result: %s", res)
        self.assertGreater(res, 0)

    def test_do_bench_profile_result_uses_linked_device_events(self):
        profiler_events = [
            FakeProfilerEvent(
                utils._DO_BENCH_PROFILE_EVENT_NAME,
                DeviceType.CPU,
                1,
                [
                    FakeProfilerEvent("aten::first", DeviceType.CPU, 2),
                    FakeProfilerEvent("aten::second", DeviceType.CPU, 3),
                    FakeProfilerEvent("cudaGraphLaunch", DeviceType.CPU, 4),
                ],
            )
        ]
        kineto_events = [
            FakeKinetoEvent(
                utils._DO_BENCH_PROFILE_EVENT_NAME,
                DeviceType.CUDA,
                0,
                11000,
                correlation_id=1,
                activity_type="gpu_user_annotation",
            ),
            FakeKinetoEvent(
                "first_kernel",
                DeviceType.CUDA,
                0,
                1000,
                linked_correlation_id=2,
            ),
            FakeKinetoEvent(
                "second_kernel",
                DeviceType.CUDA,
                10000,
                11000,
                linked_correlation_id=3,
            ),
            FakeKinetoEvent(
                "cuda_graph_kernel",
                DeviceType.CUDA,
                20000,
                21000,
                correlation_id=4,
            ),
        ]

        self.assertAlmostEqual(
            utils._get_do_bench_profile_result(
                kineto_events, profiler_events, 1, DeviceType.CUDA
            ),
            0.003,
        )

    def test_get_fused_kernel_name_windows_truncation(self):
        class FakeOrigin:
            def __init__(self, name):
                self.name = name
                self.op = "call_function"

        class FakeIRNode:
            def __init__(self, origins):
                self.origins = origins

        class FakeSchedulerNode:
            def __init__(self, origins):
                self.node = FakeIRNode(origins)

        origins = OrderedSet(
            FakeOrigin(f"some_very_long_op_name_that_exceeds_the_limit_{i}")
            for i in range(10)
        )
        node_schedule = [FakeSchedulerNode(origins)]

        # On non-Windows the full descriptive name is kept.
        with mock.patch("torch._inductor.utils.is_windows", return_value=False):
            name = utils.get_fused_kernel_name(node_schedule, "inductor_node")
            self.assertGreater(len(name), 50)

        # On Windows the name is truncated and a hash suffix is appended.
        with mock.patch("torch._inductor.utils.is_windows", return_value=True):
            name_win = utils.get_fused_kernel_name(node_schedule, "inductor_node")
            self.assertLessEqual(len(name_win), 50)
            self.assertTrue(name_win.startswith("fused_"))
            self.assertEqual(len(name_win.rsplit("_", 1)[-1]), 8)

    def test_do_bench_profile_result_requires_record_function_event(self):
        with self.assertRaisesRegex(RuntimeError, "Failed to capture"):
            utils._get_do_bench_profile_result(
                [FakeKinetoEvent("user_kernel", DeviceType.CUDA, 0, 1000)],
                [
                    FakeProfilerEvent(
                        utils._DO_BENCH_PROFILE_EVENT_NAME,
                        DeviceType.CPU,
                        1,
                    )
                ],
                1,
                DeviceType.CUDA,
            )


class TestGpuTypesDerivation(TestCase):
    """
    GPU_TYPES is no longer a hardcoded list: it is derived from the registered
    DeviceInterfaces that advertise BackendFeature.GPU. These tests pin the
    derivation to the historical list and cover the module-level __getattr__
    shim that serves it. None of them require a GPU.
    """

    # The hardcoded list GPU_TYPES had before the BackendFeature migration.
    HISTORICAL_GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]

    def test_derivation_matches_historical_list(self):
        self.assertEqual(
            sorted(utils.GPU_TYPES), sorted(self.HISTORICAL_GPU_TYPES)
        )

    def test_no_indexed_device_names(self):
        # The registry also holds "cuda:0"-style keys; GPU_TYPES is device
        # *types* only.
        for name in utils.GPU_TYPES:
            self.assertNotIn(":", name)

    def test_derivation_agrees_with_backend_features(self):
        from torch._dynamo.device_interface import get_registered_device_interfaces, BackendFeature

        expected = [
            name
            for name, iface in get_registered_device_interfaces()
            if ":" not in name and BackendFeature.GPU in iface.backend_features(None)
        ]
        self.assertEqual(sorted(utils.GPU_TYPES), sorted(expected))

    def test_from_import_sees_the_shim(self):
        # torch/testing/_internal/{common_device_type,inductor_utils}.py both do
        # `from torch._inductor.utils import GPU_TYPES` at import time; the
        # module __getattr__ must serve that form too.
        from torch._inductor.utils import GPU_TYPES

        self.assertEqual(sorted(GPU_TYPES), sorted(utils.GPU_TYPES))

    def test_repeated_access_is_stable(self):
        self.assertEqual(utils.GPU_TYPES, utils.GPU_TYPES)

    def test_unknown_attribute_still_raises(self):
        # The shim must not swallow genuine typos.
        with self.assertRaises(AttributeError):
            utils.NOT_A_REAL_ATTRIBUTE  # noqa: B018
        self.assertFalse(hasattr(utils, "NOT_A_REAL_ATTRIBUTE"))

    def test_is_gpu_zero_regression(self):
        # is_gpu() is the widest consumer of GPU_TYPES (~40 call sites).
        for name in self.HISTORICAL_GPU_TYPES:
            self.assertTrue(utils.is_gpu(name), name)
        for name in ("cpu", "tpu", "meta", "", "cuda:0"):
            self.assertFalse(utils.is_gpu(name), name)
        self.assertFalse(utils.is_gpu(None))

    def test_get_gpu_type_is_a_gpu_type(self):
        # get_gpu_type() filters GPU_TYPES by availability, defaulting to cuda.
        self.assertIn(utils.get_gpu_type(), self.HISTORICAL_GPU_TYPES)


if __name__ == "__main__":
    run_tests(device_type)

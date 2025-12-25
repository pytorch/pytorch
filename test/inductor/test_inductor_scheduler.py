# Owner(s): ["module: inductor"]

from unittest import skipIf
from unittest.mock import Mock

import torch
import torch._inductor.metrics as metrics
import torch.utils.flop_counter
from torch._dynamo.utils import counters
from torch._inductor.dependencies import Dep, ReadWrites
from torch._inductor.scheduler import BaseSchedulerNode, Scheduler
from torch._inductor.utils import fresh_inductor_cache
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    skipCUDAIf,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import IS_BIG_GPU
from torch.utils._ordered_set import OrderedSet


def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)


def get_total_flops(mode):
    return sum(v for _, v in mode.flop_counts["Global"].items())


def random_tensor(size, dtype, **kwargs):
    if dtype in [torch.half, torch.bfloat16, torch.float, torch.double]:
        return torch.randn(size, dtype=dtype, **kwargs)
    elif dtype in [torch.uint8, torch.int8, torch.short, torch.int, torch.long]:
        return torch.randint(0, 100, size, dtype=dtype, **kwargs)
    else:
        raise ValueError("Unsupported data type")


def cT(device, dtype):
    def T(*shape, requires_grad=False):
        return random_tensor(
            shape, requires_grad=requires_grad, device=device, dtype=dtype
        )

    return T


inductor_metrics_log = torch._logging.getArtifactLogger(__name__, "inductor_metrics")


def _test_cases(device, dtype):
    T = cT(device, dtype)

    def composite(x, y, z):
        tmp = torch.mm(x + 10, y / 12)
        return torch.mm(tmp, z)

    def composite_relu(x, y):
        tmp = torch.mm(x, y)
        return torch.relu(tmp)

    test_cases = [
        (torch.mm, [T(4, 5), T(5, 6)], {}),
        (torch.add, [T(4, 5), T(4, 5)], {}),
        (composite, [T(5, 4), T(4, 3), T(3, 12)], {}),
        (composite_relu, [T(5, 4), T(4, 3)], {}),
    ]
    return test_cases


class TestScheduler(TestCase):
    @dtypes(torch.float, torch.float16)
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    def test_disable_get_estimated_runtime_logging(self, device, dtype):
        if device == "cpu":
            return
        tc = _test_cases(device, dtype)
        # turn off logging of inductor metrics so that they don't get logged
        torch._logging.set_logs(inductor_metrics=False)
        metrics.reset()
        for op, example_inputs, kwargs in tc:
            comp = torch.compile(op)
            torch._dynamo.reset()
            with fresh_inductor_cache():
                comp(*example_inputs, **kwargs)
            self.assertEqual(metrics.num_bytes_accessed, 0)
            self.assertEqual(any(m[1] for m in metrics.node_runtimes), False)
            self.assertEqual(any(m[1] for m in metrics.nodes_num_elem), False)
            metrics.reset()
        torch._logging.set_logs()

    @dtypes(torch.float, torch.float16)
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @parametrize(
        "options",
        [
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON,ATEN",
            },
        ],
    )
    @torch._inductor.config.patch({"force_disable_caches": True})
    @skipIf(not IS_BIG_GPU, "we can't use Triton only as a backend for max autotune")
    def test_flop_counter_op(self, device, dtype, options):
        if device == "cpu":
            return

        tc = _test_cases(device, dtype)

        torch._logging.set_logs(inductor_metrics=True)
        for op, example_inputs, kwargs in tc:
            comp = torch.compile(op, options=options)
            # next two lines are required, otherwise the flops will be cached from previous runs of this function.
            torch._dynamo.reset()
            with fresh_inductor_cache():
                # actually run to set the counters
                comp(*example_inputs, **kwargs)
                with FlopCounterMode() as mode:
                    comp(*example_inputs, **kwargs)
            reference_flops = get_total_flops(mode)

            self.assertEqual(
                reference_flops,
                counters["inductor"]["flop_count"],
                msg=f"op = {op} reference flops = {reference_flops} != counters {counters['inductor']['flop_count']}",
            )
            if op != torch.add:
                self.assertNotEqual(reference_flops, 0, msg=f"op = {op} is 0 flops")
            counters["inductor"]["flop_count"] = 0
        torch._logging.set_logs()

    def test_fusion_prevent_too_many_reads_and_writes_prevents_fusion(self):
        """Test that fusion is prevented when unique I/O buffers exceed threshold"""
        # Setup: Create nodes with many unique I/O buffers
        # node1: reads [A, B, C], writes [D]
        # node2: reads [D, E, F], writes [G]
        # D becomes internal (node2 reads node1's write)
        # After fusion: unique I/O = {A, B, C, E, F, G} = 6 buffers
        scheduler = Mock(spec=Scheduler)
        scheduler.can_buffer_be_removed_through_fusion = Mock(return_value=False)

        node1 = self._create_mock_node(
            name="node1", reads=["A", "B", "C"], writes=["D"]
        )
        node2 = self._create_mock_node(
            name="node2", reads=["D", "E", "F"], writes=["G"]
        )

        # Execute: Check with threshold of 5 (should prevent fusion since 6 > 5)
        result = Scheduler.fusion_prevent_too_many_reads_and_writes(
            scheduler, node1, node2, threshold=5
        )

        # Assert: Fusion should be prevented (6 unique buffers > 5 threshold)
        self.assertTrue(result)

    def test_fusion_prevent_too_many_reads_and_writes_allows_fusion(self):
        """Test that fusion is allowed when intermediate buffers are removed"""
        # Setup: Create nodes where node2 reads node1's output
        # node1: reads [A, B], writes [C]
        # node2: reads [C, D], writes [E]
        # C becomes internal (node2 reads node1's write)
        # After fusion: unique I/O = {A, B, D, E} = 4 buffers
        scheduler = Mock(spec=Scheduler)
        scheduler.can_buffer_be_removed_through_fusion = Mock(return_value=False)

        node1 = self._create_mock_node(name="node1", reads=["A", "B"], writes=["C"])
        node2 = self._create_mock_node(name="node2", reads=["C", "D"], writes=["E"])

        # Execute: Check with threshold of 5 (should allow fusion since 4 <= 5)
        result = Scheduler.fusion_prevent_too_many_reads_and_writes(
            scheduler, node1, node2, threshold=5
        )

        # Assert: Fusion should be allowed (4 unique buffers <= 5 threshold)
        self.assertFalse(result)

    def _create_mock_node(self, name: str, reads: list[str], writes: list[str]) -> Mock:
        """Helper method to create a mock scheduler node with specified reads/writes"""
        node = Mock(spec=BaseSchedulerNode)
        node.get_name = Mock(return_value=name)
        node.get_nodes = Mock(return_value=[node])

        # Create mock Dep objects for reads and writes
        read_deps = OrderedSet()
        for read_name in reads:
            dep = Mock(spec=Dep)
            dep.name = read_name
            read_deps.add(dep)

        write_deps = OrderedSet()
        for write_name in writes:
            dep = Mock(spec=Dep)
            dep.name = write_name
            write_deps.add(dep)

        # Create mock ReadWrites object
        read_writes = Mock(spec=ReadWrites)
        read_writes.reads = read_deps
        read_writes.writes = write_deps

        node.read_writes = read_writes
        return node


instantiate_device_type_tests(TestScheduler, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()

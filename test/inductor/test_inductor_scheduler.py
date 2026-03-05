# Owner(s): ["module: inductor"]

from unittest import skipIf
from unittest.mock import Mock

import torch
import torch._inductor.config as inductor_config
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
    onlyCUDA,
    skipCUDAIf,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfXpu,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_BIG_GPU
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

    @skipIfXpu(
        msg="InvalidModule: Invalid SPIR-V module, "
        "https://github.com/intel/torch-xpu-ops/issues/2329"
    )
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

    @onlyCUDA
    def test_index_add_fusion_prevented(self):
        """
        Test that index_add_ (scatter with atomic_add mode) is not fused with
        subsequent reads from the same buffer, preventing read-after-write hazards.

        Regression test for: index_add_ followed by indexing was incorrectly fused,
        causing reads to occur before atomic writes completed.
        """

        def fn(f, batch):
            # Scatter: atomic writes to shared location
            f_u = f**2 + 0.00987654321
            n_batch = batch.max() + 1
            F_u_mol = torch.zeros((n_batch, f.shape[1]), device=f.device, dtype=f.dtype)
            F_u_mol.index_add_(0, batch, f_u)

            # Gather: reads from same buffer (requires synchronization)
            F_u_at_atom = F_u_mol[batch] + 1e-6
            return f_u / F_u_at_atom

        device = "cuda"
        f = torch.ones(1024, 1, device=device)
        batch = torch.zeros(1024, dtype=torch.long, device=device)

        # Eager execution (ground truth)
        eager_result = fn(f, batch)

        # Compiled execution (should match eager)
        compiled_fn = torch.compile(fn)
        compiled_result = compiled_fn(f, batch)

        # Verify results match (no fusion bug)
        self.assertTrue(
            torch.allclose(eager_result, compiled_result, rtol=1e-4, atol=1e-4),
            msg=f"index_add_ fusion bug detected: "
            f"eager={eager_result.mean().item():.6f}, "
            f"compiled={compiled_result.mean().item():.6f}",
        )

    @onlyCUDA
    def test_atomic_add_no_fusion_correctness(self):
        """
        Test that atomic_add operations produce correct results.
        """

        def fn(x, idx):
            out = torch.zeros(10, device=x.device)
            out.index_add_(0, idx, x)  # atomic_add: scatter to shared locations
            return out[idx] + 1.0  # read from same buffer: requires sync

        device = "cuda"
        x = torch.ones(5, device=device)
        idx = torch.tensor([0, 1, 0, 1, 0], device=device, dtype=torch.long)

        # Eager (correct) result
        expected = fn(x, idx)

        # Compiled result: will be wrong if fusion bug exists
        compiled_fn = torch.compile(fn)
        torch._dynamo.reset()
        with fresh_inductor_cache():
            result = compiled_fn(x, idx)

        # This test will FAIL without the fusion prevention fix
        self.assertTrue(
            torch.allclose(expected, result),
            msg=f"Fusion bug detected! Expected {expected}, got {result}",
        )


class TestScoreFusionMemory(TestCase):
    """
    Tests for _score_fusion_memory_by_buffer_overlap.

    These tests validate the fusion scoring logic that determines when nodes
    should be fused together based on their memory access patterns.

    Key scenarios:
    1. Exact matches: read/write has exact matches → should fuse (1 kernel)
    2. Large overlap (split/cat): reads on different offset but overlap is huge
       → should fuse because the benefit is large (1 kernel)
    3. Small overlap: reads on different offset but overlap is small → don't fuse (2 kernels)
    """

    @skipIf(not HAS_GPU, "GPU not available")
    @inductor_config.patch("score_fusion_memory_threshold", 1)
    def test_exact_same_reads_should_fuse(self) -> None:
        """
        Case 1: Exact matches in read/write → should fuse into 1 kernel.

        Two operations reading from the exact same input tensor should be
        fused together since they can share the data read from memory.
        """

        def exact_reads(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Both operations read the exact same input
            out1 = x * 2
            out2 = x + 1
            return out1, out2

        torch._dynamo.reset()
        metrics.reset()

        x = torch.randn(8, 512, device=GPU_TYPE, dtype=torch.float16)

        compiled_fn = torch.compile(exact_reads, backend="inductor", fullgraph=True)
        out1_eager, out2_eager = exact_reads(x)
        out1_compiled, out2_compiled = compiled_fn(x)

        self.assertTrue(torch.allclose(out1_eager, out1_compiled, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(out2_eager, out2_compiled, atol=1e-3, rtol=1e-3))
        # Should fuse into 1 kernel since both ops read exact same buffer
        self.assertEqual(metrics.generated_kernel_count, 1)

    @skipIf(not HAS_GPU, "GPU not available")
    @inductor_config.patch("score_fusion_memory_threshold", 1)
    def test_split_cat_large_overlap_should_fuse(self) -> None:
        """
        Case 2: Reads on different offset but overlap is huge (split/cat) → should fuse into 1 kernel.

        Split operations read from the same input buffer at different offsets.
        Since the overlap is large (same underlying buffer), fusing these
        operations together saves reads and kernel launches.
        """

        def split_and_process(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            s1, s2, s3, s4 = torch.split(x, x.shape[-1] // 4, dim=-1)
            out1 = torch.cat([s4, s3], dim=-1)
            out2 = torch.cat([s2, s1], dim=-1)
            return out1, out2

        torch._dynamo.reset()
        metrics.reset()

        x = torch.randn(8, 512, device=GPU_TYPE, dtype=torch.float16)

        compiled_fn = torch.compile(
            split_and_process, backend="inductor", fullgraph=True
        )
        out1_eager, out2_eager = split_and_process(x)
        out1_compiled, out2_compiled = compiled_fn(x)

        self.assertTrue(torch.allclose(out1_eager, out1_compiled, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(out2_eager, out2_compiled, atol=1e-3, rtol=1e-3))
        # Should fuse into 1 kernel since all ops read from the same underlying buffer
        self.assertEqual(metrics.generated_kernel_count, 1)

    @skipIf(not HAS_GPU, "GPU not available")
    @inductor_config.patch("score_fusion_memory_threshold", 1)
    def test_partial_overlap_below_threshold(self) -> None:
        """
        Case 3: Partial overlap below the 0.5 threshold → should NOT fuse (2 kernels).

        Similar to test_split_cat_large_overlap_should_fuse, but each operation
        also reads from a separate large tensor, making the shared buffer portion
        less than 50% of total reads.

        Example scenario:
        - Split x into 4 slices: s1, s2, s3, s4 (each 25% of x)
        - op1 reads: s1 (from x, ~25%) + y (separate tensor, ~75%) → total 100%
        - op2 reads: s2 (from x, ~25%) + z (separate tensor, ~75%) → total 100%
        - Common buffer is x, but each op only reads 25% of their total from x
        - overlap_ratio = 25% / 100% = 0.25 < 0.5 threshold → score = 0
        - Result: 2 separate kernels (not fused)
        """

        def partial_overlap_split(
            x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Split x into 4 parts, use different slices in each output
            s1, s2, _, _ = torch.split(x, x.shape[-1] // 4, dim=-1)
            # op1 reads: s1 (small slice of x) + y (large separate tensor)
            # op2 reads: s2 (small slice of x) + z (large separate tensor)
            # The slices s1 and s2 come from the same buffer x,
            # but each is only ~25% of total reads for that op
            out1 = torch.cat([s1, y, y, y], dim=-1)
            out2 = torch.cat([s2, z, z, z], dim=-1)
            return out1, out2

        torch._dynamo.reset()
        metrics.reset()

        # x is split into 4 parts (each 128 elements)
        # y and z are 3x larger (384 elements each)
        # So each op reads: 128 (from x slice) + 384 (from y or z) = 512 total
        # overlap_ratio = 128 / 512 = 0.25 < 0.5 threshold
        x = torch.randn(8, 512, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(8, 128, device=GPU_TYPE, dtype=torch.float16)
        z = torch.randn(8, 128, device=GPU_TYPE, dtype=torch.float16)

        compiled_fn = torch.compile(
            partial_overlap_split, backend="inductor", fullgraph=True
        )
        out1_eager, out2_eager = partial_overlap_split(x, y, z)
        out1_compiled, out2_compiled = compiled_fn(x, y, z)

        self.assertTrue(torch.allclose(out1_eager, out1_compiled, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(out2_eager, out2_compiled, atol=1e-3, rtol=1e-3))
        # Should NOT fuse (2 kernels) because overlap_ratio = 0.25 < 0.5 threshold
        # The _score_fusion_memory_by_buffer_overlap returns 0 for this case
        self.assertEqual(metrics.generated_kernel_count, 2)


instantiate_device_type_tests(TestScheduler, globals(), allow_xpu=True)
instantiate_device_type_tests(TestScoreFusionMemory, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()

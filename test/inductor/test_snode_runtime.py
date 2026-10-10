# Owner(s): ["module: inductor"]

import contextlib
from unittest import skipIf

import torch
import torch.distributed as dist
from torch._inductor import config, metrics
from torch._inductor.comm_analysis import estimate_nccl_collective_runtime
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import is_collective
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


aten = torch.ops.aten
c10d = torch.ops.c10d_functional
_c10d = torch.ops._c10d_functional


def compile_but_use_eager(gm, example_inputs):
    def inner_compile(gm, *args, **kwargs):
        compile_fx_inner(gm, *args, **kwargs)
        return gm

    return compile_fx(gm, example_inputs, inner_compile=inner_compile)


def calculate_runtime(f, *args) -> float:
    """
    Assumes all inputs are fp32
    """
    metrics.reset()
    torch._logging.set_logs(inductor_metrics=True)
    torch.compile(f, backend=compile_but_use_eager)(*args)
    print(metrics.node_runtimes)

    ret = 0.0
    for pair in metrics.node_runtimes:
        ret += pair[1]

    torch._logging.set_logs()
    return ret


def T(*size, device, dtype=torch.float32, grad=False) -> torch.Tensor:
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


class TestCase(InductorTestCase):
    """
    Helper methods to compare runtime estimate against 0. Since this estimate is hardware dependent,
    stronger comparisons may fail depending on the host's specs.

    atol/rtol must be provided explicitly with each call, since precision/rel_tol overrides are not always utilized
    """

    def setUp(self):
        super().setUp()
        # These tests check metrics.node_runtimes and we don't save / restore
        # those in the FX graph cache.
        self._test_snode_stack = contextlib.ExitStack()
        self._test_snode_stack.enter_context(
            config.patch({"fx_graph_remote_cache": False})
        )

    def tearDown(self):
        self._test_snode_stack.close()
        super().tearDown()

    def assertZero(self, x: float):
        if not isinstance(x, float):
            raise AssertionError(f"Expected float, got {type(x)}")
        super().assertEqual(x, 0.0, atol=0, rtol=0)

    def assertNotZero(self, x):
        if not isinstance(x, float):
            raise AssertionError(f"Expected float, got {type(x)}")
        super().assertNotEqual(x, 0.0, atol=0, rtol=0)


class UnsupportedTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_no_op(self, device):
        def f(a):
            return a

        inp = (T(10, 10, device=device),)
        self.assertZero(calculate_runtime(f, *inp))


class UnsupportedTestsGeneric(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_no_cuda(self):
        def f(a):
            return a

        inp = (torch.randn((10, 10), device="cpu"),)
        self.assertZero(calculate_runtime(f, *inp))


class ComputeBoundedTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_conv1d(self, device):
        def f(x, y):
            return torch.nn.functional.conv1d(x, y)

        inp = (T(33, 16, 30, device=device), T(20, 16, 5, device=device))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv2d(self, device):
        def f(x, y):
            return torch.nn.functional.conv2d(x, y, padding=1)

        inp = (T(8, 4, 3, 3, device=device), T(1, 4, 5, 5, device=device))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv2d_transpose(self, device):
        def f(x, y):
            return torch.nn.functional.conv_transpose2d(x, y, padding=1)

        inp = (T(8, 1, 1, 1, device=device), T(1, 4, 5, 5, device=device))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv3d(self, device):
        def f(x, y):
            return torch.nn.functional.conv3d(x, y)

        inp = (T(20, 16, 50, 10, 20, device=device), T(33, 16, 3, 3, 3, device=device))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_mm(self, device):
        def f(a, b):
            return torch.mm(a, b)

        inp = (
            T(10, 10, device=device),
            T(10, 10, device=device),
        )
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_addmm(self, device):
        def f(a, b, c):
            return torch.addmm(a, b, c)

        inp = (
            T(10, 10, device=device),
            T(10, 10, device=device),
            T(10, 10, device=device),
        )
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_bmm(self, device):
        def f(a, b):
            return torch.bmm(a, b)

        inp = (
            T(10, 10, 10, device=device),
            T(10, 10, 10, device=device),
        )
        self.assertNotZero(calculate_runtime(f, *inp))


class MemoryBoundedTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_relu(self, device):
        def f(a):
            return torch.nn.functional.relu(a)

        inp = (T(10, 10, device=device),)
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_horizontal_reduction_pointwise(self, device):
        def f(a):
            b = a.sum(dim=1)
            c = a.cos()
            return b, c

        inp = (T(10, 10, device=device),)
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_pointwise(self, device):
        def f(x):
            return x.cos()

        inp = (T(10, device=device),)
        self.assertNotZero(calculate_runtime(f, *inp))

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic(self, device):
        def f(x):
            return x.cos()

        inp = (T(10, device=device),)
        self.assertNotZero(calculate_runtime(f, *inp))


class InputDistanceTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def _get_snodes(self, f, *args):
        metrics.reset()
        torch._logging.set_logs(inductor_metrics=True)
        torch.compile(f, backend=compile_but_use_eager)(*args)
        torch._logging.set_logs()
        return [snode for snode, _ in metrics.nodes_num_elem]

    def test_chain_with_reduction(self, device):
        """
        input -> sum (depth 0) -> sum (depth 1)
        Reductions prevent full fusion, giving us distinct depth levels.
        """

        def f(x):
            a = x.sum(dim=-1)
            return a.sum(dim=-1)

        snodes = self._get_snodes(f, T(10, 10, 10, device=device))
        all_min = [s.min_input_distance for s in snodes]
        all_max = [s.max_input_distance for s in snodes]
        self.assertEqual(min(all_min), 0)
        self.assertEqual(max(all_max), 1)

    def test_fused_node_depth_range(self, device):
        """
        A reduction fused with its pointwise epilogue should have
        min_input_distance=0 and max_input_distance=1.
        """

        def f(x):
            a = x.sum(dim=-1)
            return a.cos()

        snodes = self._get_snodes(f, T(10, 10, device=device))
        # The reduction and pointwise get fused
        self.assertEqual(len(snodes), 1)
        self.assertEqual(snodes[0].min_input_distance, 0)
        self.assertEqual(snodes[0].max_input_distance, 1)

    def test_extern_kernel_chain(self, device):
        """
        mm (depth 0, extern) -> cos+sum fused (depth 1)
        """

        def f(a, b):
            c = torch.mm(a, b)
            d = c.cos()
            return d.sum(dim=-1)

        snodes = self._get_snodes(f, T(10, 10, device=device), T(10, 10, device=device))
        all_min = [s.min_input_distance for s in snodes]
        all_max = [s.max_input_distance for s in snodes]
        self.assertEqual(min(all_min), 0)
        self.assertEqual(max(all_max), 1)

    def test_foreach_basic(self, device):
        """
        foreach_add on graph inputs should have depth 0.
        """

        def f(xs, ys):
            return torch._foreach_add(xs, ys)

        xs = [T(10, device=device), T(20, device=device)]
        ys = [T(10, device=device), T(20, device=device)]
        snodes = self._get_snodes(f, xs, ys)
        for s in snodes:
            self.assertEqual(s.min_input_distance, 0)
            self.assertEqual(s.max_input_distance, 0)

    def test_foreach_after_extern(self, device):
        """
        mm (extern, depth 0) -> foreach_add (depth 1)
        The extern kernel creates a fusion barrier so the foreach
        has a real dependency chain.
        """

        def f(a, b, ys):
            c = torch.mm(a, b)
            return torch._foreach_add([c, c], ys)

        snodes = self._get_snodes(
            f,
            T(10, 10, device=device),
            T(10, 10, device=device),
            [T(10, 10, device=device), T(10, 10, device=device)],
        )
        all_min = [s.min_input_distance for s in snodes]
        all_max = [s.max_input_distance for s in snodes]
        self.assertEqual(min(all_min), 0)
        self.assertGreaterEqual(max(all_max), 1)


@skipIf(not dist.is_available(), "requires distributed")
class TestCommAnalysis(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    WORLD_SIZE: int = 8
    RANKS = list(range(8))

    def _verify_runtime_estimation(self, fn, inps):
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.WORLD_SIZE, store=store
        )
        try:
            metrics.reset()
            torch._logging.set_logs(inductor_metrics=True)
            torch.compile(fn)(*inps)
            found_collective = False
            for snode, runtime in metrics.node_runtimes:
                if not is_collective(snode.node):
                    continue
                found_collective = True
                # Inductor swallows errors from snode runtime estimations.
                # We call estimate_nccl_collective_runtime in a white-box
                # fashion here so potential issues can be surfaced in tests.
                est = estimate_nccl_collective_runtime(snode.node)
                self.assertNotZero(est)
                # Also make sure estimate_nccl_collective_runtime works
                # correctly in inductor.
                self.assertNotZero(runtime)
            # Make sure a collective kernel is found in graph
            self.assertTrue(found_collective)
            torch._logging.set_logs()
        finally:
            dist.destroy_process_group()

    def test_legacy_all_reduce(self, device):
        def fn(x):
            r = c10d.all_reduce(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return c10d.wait_tensor(r)

        inp = T(10, 10, device=device)
        self._verify_runtime_estimation(fn, (inp,))

    def test_legacy_all_reduce_coalesced(self, device):
        def fn(x):
            rs = c10d.all_reduce_coalesced(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return [c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10, device=device), T(15, 15, device=device)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_legacy_all_gather_into_tensor_coalesced(self, device):
        def fn(x):
            rs = c10d.all_gather_into_tensor_coalesced(
                x,
                "",
                self.RANKS,
                self.WORLD_SIZE,
            )
            return [c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10, device=device), T(15, 15, device=device)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_reduce(self, device):
        def fn(x):
            r = _c10d.all_reduce(x, "sum", "0")
            return _c10d.wait_tensor(r)

        inp = T(10, 10, device=device)
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_reduce_coalesced(self, device):
        def fn(x):
            rs = _c10d.all_reduce_coalesced(x, "sum", "0")
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10, device=device), T(15, 15, device=device)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_gather_into_tensor(self, device):
        def fn(x):
            rs = _c10d.all_gather_into_tensor(
                x,
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = T(10, 10, device=device)
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_gather_into_tensor_coalesced(self, device):
        def fn(x):
            rs = _c10d.all_gather_into_tensor_coalesced(
                x,
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10, device=device), T(15, 15, device=device)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_reduce_scatter_tensor(self, device):
        def fn(x):
            rs = _c10d.reduce_scatter_tensor(
                x,
                "sum",
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = T(self.WORLD_SIZE, 10, device=device)
        self._verify_runtime_estimation(fn, (inp,))

    def test_reduce_scatter_tensor_coalesced(self, device):
        def fn(x):
            rs = _c10d.reduce_scatter_tensor_coalesced(
                x,
                "sum",
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [
            T(self.WORLD_SIZE, 10, device=device),
            T(self.WORLD_SIZE, 15, device=device),
        ]
        self._verify_runtime_estimation(fn, (inp,))


instantiate_device_type_tests(
    UnsupportedTests, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(
    ComputeBoundedTests, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(
    MemoryBoundedTests, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(
    InputDistanceTests, globals(), except_for="cpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestCommAnalysis, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests(needs="filelock")

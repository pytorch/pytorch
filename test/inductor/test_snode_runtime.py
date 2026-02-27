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
from torch.testing._internal.common_device_type import expectedFailureXPU
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


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


DEVICE = GPU_TYPE


def T(*size, dtype=torch.float32, device=DEVICE, grad=False) -> torch.Tensor:
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


class TestCase(InductorTestCase):
    device = DEVICE

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
    device = DEVICE

    def test_no_op(self):
        def f(a):
            return a

        inp = (T(10, 10),)
        self.assertZero(calculate_runtime(f, *inp))

    def test_no_cuda(self):
        def f(a):
            return a

        inp = (torch.randn((10, 10), device="cpu"),)
        self.assertZero(calculate_runtime(f, *inp))


class ComputeBoundedTests(TestCase):
    device = DEVICE

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_conv1d(self):
        def f(x, y):
            return torch.nn.functional.conv1d(x, y)

        inp = (T(33, 16, 30), T(20, 16, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_conv2d(self):
        def f(x, y):
            return torch.nn.functional.conv2d(x, y, padding=1)

        inp = (T(8, 4, 3, 3), T(1, 4, 5, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_conv2d_transpose(self):
        def f(x, y):
            return torch.nn.functional.conv_transpose2d(x, y, padding=1)

        inp = (T(8, 1, 1, 1), T(1, 4, 5, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_conv3d(self):
        def f(x, y):
            return torch.nn.functional.conv3d(x, y)

        inp = (T(20, 16, 50, 10, 20), T(33, 16, 3, 3, 3))
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_mm(self):
        def f(a, b):
            return torch.mm(a, b)

        inp = (
            T(10, 10),
            T(10, 10),
        )
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_addmm(self):
        def f(a, b, c):
            return torch.addmm(a, b, c)

        inp = (
            T(10, 10),
            T(10, 10),
            T(10, 10),
        )
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_bmm(self):
        def f(a, b):
            return torch.bmm(a, b)

        inp = (
            T(10, 10, 10),
            T(10, 10, 10),
        )
        self.assertNotZero(calculate_runtime(f, *inp))


class MemoryBoundedTests(TestCase):
    device = DEVICE

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_relu(self):
        def f(a):
            return torch.nn.functional.relu(a)

        inp = (T(10, 10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_horizontal_reduction_pointwise(self):
        def f(a):
            b = a.sum(dim=1)
            c = a.cos()
            return b, c

        inp = (T(10, 10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    def test_pointwise(self):
        def f(x):
            return x.cos()

        inp = (T(10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    # lack of profiler on XPU
    @expectedFailureXPU
    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic(self):
        def f(x):
            return x.cos()

        inp = (T(10),)
        self.assertNotZero(calculate_runtime(f, *inp))


@skipIf(not dist.is_available(), "requires distributed")
class TestCommAnalysis(TestCase):
    device = DEVICE

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

    def test_legacy_all_reduce(self):
        def fn(x):
            r = c10d.all_reduce(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return c10d.wait_tensor(r)

        inp = T(10, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_legacy_all_reduce_coalesced(self):
        def fn(x):
            rs = c10d.all_reduce_coalesced(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return [c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_legacy_all_gather_into_tensor_coalesced(self):
        def fn(x):
            rs = c10d.all_gather_into_tensor_coalesced(
                x,
                "",
                self.RANKS,
                self.WORLD_SIZE,
            )
            return [c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_reduce(self):
        def fn(x):
            r = _c10d.all_reduce(x, "sum", "0")
            return _c10d.wait_tensor(r)

        inp = T(10, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_reduce_coalesced(self):
        def fn(x):
            rs = _c10d.all_reduce_coalesced(x, "sum", "0")
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_gather_into_tensor(self):
        def fn(x):
            rs = _c10d.all_gather_into_tensor(
                x,
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = T(10, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_gather_into_tensor_coalesced(self):
        def fn(x):
            rs = _c10d.all_gather_into_tensor_coalesced(
                x,
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_reduce_scatter_tensor(self):
        def fn(x):
            rs = _c10d.reduce_scatter_tensor(
                x,
                "sum",
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = T(self.WORLD_SIZE, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_reduce_scatter_tensor_coalesced(self):
        def fn(x):
            rs = _c10d.reduce_scatter_tensor_coalesced(
                x,
                "sum",
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(self.WORLD_SIZE, 10), T(self.WORLD_SIZE, 15)]
        self._verify_runtime_estimation(fn, (inp,))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")

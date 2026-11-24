# Owner(s): ["module: cuda graphs"]

import functools

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._inductor.cudagraph_trees as cudagraph_trees
from torch._dynamo.testing import same
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIf,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    TEST_CUDA_GRAPH,
    TEST_XPU,
)
from torch.testing._internal.logging_utils import logs_to_string


def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def assert_aot_autograd_counter(ok=True):
    def deco(f):
        @functools.wraps(f)
        def wrap(self, *args, **kwargs):
            torch._dynamo.utils.counters.clear()
            r = f(self, *args, **kwargs)
            c_ok = torch._dynamo.utils.counters["aot_autograd"]["ok"]
            c_not_ok = torch._dynamo.utils.counters["aot_autograd"]["not_ok"]
            if ok:
                self.assertGreater(c_ok, 0)
                self.assertEqual(c_not_ok, 0)
            else:
                self.assertEqual(c_ok, 0)
                self.assertGreater(c_not_ok, 0)
            return r

        return wrap

    return deco


def patch_all(ok=True):
    return composed(
        torch._dynamo.config.patch(
            verify_correctness=True, automatic_dynamic_shapes=True
        ),
        assert_aot_autograd_counter(ok),
    )


N_ITERS = 5


class TestAotCudagraphs(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @patch_all()
    def test_basic(self, device):
        def model(x, y):
            return (x + y) * y

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn(3, device=device)
        fn(x, y)

    @patch_all()
    def test_dtoh(self, device):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn(3, device=device)
        fn(x, y)

    @torch._inductor.config.patch({"graph_partition": True, "triton.cudagraphs": True})
    def test_cpu_input_skips_despite_inductor_graph_partition(self, device):
        # This backend never partitions, so Inductor's graph_partition must not
        # make it tolerate a cpu input: the capture would fail.
        def fn(x, y):
            return x + 1, y + 2

        compiled = torch.compile(fn, backend="cudagraphs")
        log_stream, ctx = logs_to_string(
            "torch._inductor.cudagraph_utils", "cudagraphs"
        )
        with ctx():
            for i in range(3):
                x = torch.full((4,), float(i), device=device)
                y = torch.full((4,), float(10 * i))
                self.assertEqual(compiled(x, y), fn(x, y))
        FileCheck().check("skipping cudagraphs due to cpu device (arg1_1)").run(
            log_stream.getvalue()
        )
        self.assertEqual(torch._dynamo.utils.counters["inductor"]["cudagraph_skips"], 1)

    @patch_all()
    def test_htod(self, device):
        def model(x, y):
            a = x + y
            return a * 3

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn((), device="cpu")
        fn(x, y)

    def test_mutate_input(self, device):
        def model(x, y):
            y.add_(3)
            return x * y

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertTrue(same(y, y_orig + 3))
                    loss.backward()

        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn(3, device=device)
        fn(x, y)

    @patch_all()
    def test_mutate_constant(self, device):
        def model(x, y):
            c = torch.tensor(1)
            c.add_(2)
            return x * y * 0 + c

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(x, y).sum()
                    self.assertTrue(same(loss, torch.tensor(3.0, device=device)))
                    loss.backward()

        x = torch.randn(1, device=device, requires_grad=True)
        y = torch.randn(1, device=device)
        fn(x, y)

    @patch_all()
    def test_factory(self, device):
        def model(y):
            x = torch.zeros(3, device=device)
            x.add_(3)
            return x * y

        @torch.compile(backend="cudagraphs")
        def fn(y):
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(y).sum()
                    loss.backward()

        y = torch.randn(3, device=device, requires_grad=True)
        fn(y)

    @patch_all()
    def test_mutated_metadata(self, device):
        # more tortured example at
        # https://github.com/pytorch/pytorch/issues/81385
        def model(x):
            x = x.clone()
            x.resize_(20)
            x.fill_(2)
            return x

        @torch.compile(backend="cudagraphs")
        def fn(x):
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device=device)))

        x = torch.empty(0, device=device)
        fn(x)

    @patch_all()
    def test_dead_fill(self, device):
        def model(x):
            x = x.clone()
            y = x[0:0]
            x.fill_(2)
            y.fill_(3)
            return x, y

        @torch.compile(backend="cudagraphs")
        def fn(x):
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx, ry = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device=device)))
                    self.assertTrue(same(ry, torch.empty(0, device=device)))

        x = torch.empty(20, device=device)
        fn(x)

    @skipCUDAIf(not TEST_CUDA_GRAPH, "cuda graph test is skipped")
    def test_inference_mode_does_not_need_mark_step_begin(self, device):
        # This tests that torch.inference_mode() removes the need for
        # the user to call torch.compiler.mark_step_begin()

        # reset to make sure that warned_functions is empty
        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        def foo(x):
            with torch.inference_mode():
                return x.sin()

        x = torch.randn(4, device=device, requires_grad=False)

        compiled_foo = torch.compile(foo, backend="cudagraphs")

        # First call runs eager, second captures and replays, rest
        # should just replay.
        for _ in range(4):
            out = compiled_foo(x)
            self.assertTrue(same(out, foo(x)))

        manager = cudagraph_trees.get_manager(x.device.index)
        self.assertTrue(
            len(manager.warned_functions) == 0,
            "Replaying inference workload should not warn about repeated graph captures",
        )

    def test_compile_during_capture_errors(self, device):
        # https://github.com/pytorch/pytorch/issues/185074
        from torch._dynamo.exc import TorchRuntimeError

        def fn(x):
            return x + 1

        comp_fn = torch.compile(backend="eager")(fn)

        x = torch.randn(10, device=device)

        capture = (
            torch.cuda.graph(torch.cuda.CUDAGraph())
            if self.device_type == "cuda"
            else torch.accelerator.Graph()
        )
        # Should raise TorchRuntimeError when trying to compile during capture
        with (
            capture,
            self.assertRaisesRegex(
                TorchRuntimeError,
                rf"cannot JIT compile during {self.device_type.upper()} graph capture",
            ),
        ):
            comp_fn(x)

        self.assertEqual(comp_fn(x), fn(x))


instantiate_device_type_tests(
    TestAotCudagraphs, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_CUDA_GRAPH and not TEST_XPU:
        import sys

        sys.exit(0)

    run_tests()

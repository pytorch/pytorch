# Owner(s): ["module: cuda"]
# ruff: noqa: F821, F841

import sys
import warnings

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    skipIfRocm,
    TEST_CUDA,
    TestCase,
)


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


def _warmup_op(op, n=3):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(n):
            op()


def _capture_graph(op, **graph_kwargs):
    g = torch.cuda.CUDAGraph()
    _warmup_op(op)
    graph_kwargs.setdefault("check_input_liveness", True)
    with torch.cuda.graph(g, **graph_kwargs):
        out = op()
    return g, out


@skipIfRocm(msg="input liveness checking is only supported on NVIDIA CUDA")
class TestCUDAGraphDebugInputs(TestCase):
    def test_nothing_dead_ok(self):
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        g.replay()
        self.assertEqual(y, x * 2)

    def test_dead_input_tensor(self):
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        del x
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_dead_intermediate_tensors_ok(self):
        x = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, check_input_liveness=True):
            tmp = x * 2
            y = tmp + 1
        del tmp
        g.replay()
        self.assertEqual(y, x * 2 + 1)
        del y
        g.replay()

    def test_model_weight_dead(self):
        class SimpleMLP(nn.Module):
            def __init__(self, width, depth):
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(width, width) for _ in range(depth)]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x

        model = SimpleMLP(width=64, depth=5).cuda()
        x = torch.randn(8, 64, device="cuda")
        g = torch.cuda.CUDAGraph()
        _warmup_op(lambda: model(x))
        with torch.no_grad(), torch.cuda.graph(g, check_input_liveness=True):
            y = model(x)
        g.replay()
        # self.assertEqual(y, model(x)) make code change to retrigger CI
        del model.layers[2].weight
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_model_training_weight_dead(self):
        model = nn.Linear(32, 16).cuda()
        x = torch.randn(8, 32, device="cuda")
        static_grad = torch.randn(8, 16, device="cuda")
        with warnings.catch_warnings():
            # First cuBLAS backward triggers "no current CUDA context" warning
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(3):
                model(x).backward(static_grad)
        g = torch.cuda.CUDAGraph()
        model.zero_grad()
        with torch.cuda.graph(g, check_input_liveness=True):
            out = model(x)
            out.backward(static_grad)
        g.replay()
        # out's grad_fn retains weight as a saved tensor from forward
        del out
        del model.weight
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_view_tensor(self):
        base = torch.randn(100, device="cuda")
        view = base[10:50]
        g, y = _capture_graph(lambda: view * 2)
        del view
        g.replay()
        del base
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_nested_structure_inputs(self):
        xs = [torch.randn(10, device="cuda") for _ in range(3)]
        ys = [torch.randn(10, device="cuda") for _ in range(3)]
        g, result = _capture_graph(lambda: torch._foreach_add(xs, ys))
        del ys[1]
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_empty_tensor_not_tracked(self):
        x = torch.randn(100, device="cuda")
        empty = torch.empty(0, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, check_input_liveness=True):
            y = x * 2
            z = empty.view(0)
        self.assertEqual(empty.data_ptr(), 0)
        self.assertNotIn(empty.data_ptr(), g._tracker._external_inputs)
        del empty
        g.replay()

    def test_non_capturing_stream_inputs_not_tracked(self):
        capture_stream = torch.cuda.Stream()
        other_stream = torch.cuda.Stream()
        x = torch.randn(100, device="cuda")
        y = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(
                g, capture_error_mode="relaxed", check_input_liveness=True
            ):
                result = x * 2
                with torch.cuda.stream(other_stream):
                    other_result = y * 3
        self.assertIn(x.data_ptr(), g._tracker._external_inputs)
        self.assertNotIn(y.data_ptr(), g._tracker._external_inputs)
        del y
        g.replay()
        self.assertEqual(result, x * 2)

    def test_shared_pool_tensor_not_tracked(self):
        pool = torch.cuda.graph_pool_handle()
        x = torch.randn(100, device="cuda")
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1, pool=pool):
            y = x + 1
            w = torch.randn(100, device="cuda")
        external_input = torch.randn(100, device="cuda")
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2, pool=pool, check_input_liveness=True):
            z = y * 2 + external_input
            p = w + 1
        del y
        g1.replay()
        g2.replay()
        self.assertEqual(z, (x + 1) * 2 + external_input)
        del w
        g1.replay()
        g2.replay()

    def test_alive_storage_dead_tensor_ok(self):
        a = torch.randn(100, device="cuda")
        b = torch.empty(100, device="cuda")
        b.set_(a.untyped_storage())
        g, y = _capture_graph(lambda: a * 2)
        del a
        g.replay()
        self.assertEqual(y, b * 2)
        del b
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_dead_storage_raises(self):
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        x.set_(torch.randn(100, device="cuda").untyped_storage())
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_resize_larger_detected(self):
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        x.resize_(100000)
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_pinned_memory_tensor_tracked(self):
        pinned = torch.randn(100).pin_memory()
        cuda_dest = torch.empty(100, device="cuda")
        g = torch.cuda.CUDAGraph()

        def op():
            cuda_dest.copy_(pinned, non_blocking=True)
            return cuda_dest * 2

        with torch.cuda.graph(g, check_input_liveness=True):
            cuda_dest.copy_(pinned, non_blocking=True)
            y = cuda_dest * 2
        self.assertIn(pinned.data_ptr(), g._tracker._external_inputs)
        self.assertIn(cuda_dest.data_ptr(), g._tracker._external_inputs)
        # This should be OK
        g.replay()
        del pinned
        # This should raise
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_recapture_without_clears_stale_state(self):
        g = torch.cuda.CUDAGraph()
        x = torch.randn(32, device="cuda")
        with torch.cuda.graph(g, check_input_liveness=True):
            y = x * 2
        del x
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()
        g.reset()
        x2 = torch.randn(32, device="cuda")
        with torch.cuda.graph(g, check_input_liveness=True):
            y2 = x2 * 3
        g.replay()

    def test_higher_order_op_during_capture(self):
        from torch._higher_order_ops.wrap import wrap

        def inner_fn(x):
            return x * 2

        x = torch.randn(32, device="cuda")
        g, y = _capture_graph(lambda: wrap(inner_fn, x))
        g.replay()
        self.assertEqual(y, x * 2)


@skipIfRocm(msg="input liveness checking is only supported on NVIDIA CUDA")
class TestCUDAGraphDebugBacktraces(TestCase):
    def test_error_message_contains_all_tracebacks(self):
        a = torch.randn(100, device="cuda")
        b = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, check_input_liveness=True):
            y = a + b

        def delete_tensors():
            nonlocal a, b
            del a, b

        delete_tensors()
        with self.assertRaises(RuntimeError) as ctx:
            g.replay()
        error_msg = str(ctx.exception).lower()
        self.assertIn("2 dead tensor", error_msg)
        self.assertIn("creation traceback (python)", error_msg)
        self.assertIn("creation traceback (c++)", error_msg)
        self.assertIn("deletion traceback (python)", error_msg)
        self.assertIn("delete_tensors", error_msg)
        self.assertIn("replay traceback (python)", error_msg)


@skipIfRocm(msg="input liveness checking is only supported on NVIDIA CUDA")
class TestCUDAGraphDebugExternalOps(TestCase):
    def test_custom_autograd_function(self):
        from torch.autograd import Function

        class ScaleOp(Function):
            @staticmethod
            def forward(ctx, x, scale):
                ctx.save_for_backward(scale)
                return x * scale

            @staticmethod
            def backward(ctx, grad):
                (scale,) = ctx.saved_tensors
                return grad * scale, None

        x = torch.randn(100, device="cuda")
        scale = torch.tensor(2.0, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g, check_input_liveness=True):
            y = ScaleOp.apply(x, scale)
        del scale
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_torch_library_custom_op(self):
        @torch.library.custom_op("test_cuda_graph_debug::scale", mutates_args=())
        def scale_op(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return x * scale

        @scale_op.register_fake
        def _(x, scale):
            return torch.empty_like(x)

        x = torch.randn(100, device="cuda")
        scale = torch.tensor(3.0, device="cuda")
        g, y = _capture_graph(lambda: scale_op(x, scale))
        del scale
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()


if __name__ == "__main__":
    run_tests()

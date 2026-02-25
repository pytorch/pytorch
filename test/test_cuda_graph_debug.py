# Owner(s): ["module: cuda"]
# ruff: noqa: F821, F841

import gc
import sys
import warnings

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


def _sync_and_gc_collect():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _warmup_op(op, n=3):
    for _ in range(n):
        op()
    torch.cuda.synchronize()


class TestCUDAGraphDebugInputs(TestCase):
    def test_nothing_dead_ok(self):
        x = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()

        _warmup_op(lambda: x * 2)

        with torch.cuda.graph(g, check_input_liveness=True):
            y = x * 2

        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(y, x * 2)

    def test_dead_input_tensor(self):
        x = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()

        _warmup_op(lambda: x * 2)

        with torch.cuda.graph(g, check_input_liveness=True):
            y = x * 2

        del x
        _sync_and_gc_collect()

        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_dead_intermediate_tensors_ok(self):
        x = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()

        def op():
            tmp = x * 2
            return tmp + 1

        _warmup_op(op)

        with torch.cuda.graph(g, check_input_liveness=True):
            tmp = x * 2
            y = tmp + 1

        del tmp
        _sync_and_gc_collect()

        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(y, x * 2 + 1)

        del y
        _sync_and_gc_collect()

        g.replay()
        torch.cuda.synchronize()

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
        torch.cuda.synchronize()
        self.assertEqual(y, model(x))

        del model.layers[2].weight
        _sync_and_gc_collect()

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
        torch.cuda.synchronize()

        # out's grad_fn retains weight as a saved tensor from forward
        del out
        del model.weight
        _sync_and_gc_collect()

        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_view_tensor(self):
        base = torch.randn(100, device="cuda")
        view = base[10:50]
        g = torch.cuda.CUDAGraph()

        _warmup_op(lambda: view * 2)

        with torch.cuda.graph(g, check_input_liveness=True):
            y = view * 2

        del view
        _sync_and_gc_collect()

        # should be ok, the base tensor's memory is still alive
        g.replay()
        torch.cuda.synchronize()

        del base
        _sync_and_gc_collect()

        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_nested_structure_inputs(self):
        xs = [torch.randn(10, device="cuda") for _ in range(3)]
        ys = [torch.randn(10, device="cuda") for _ in range(3)]
        g = torch.cuda.CUDAGraph()

        def op():
            return torch._foreach_add(xs, ys)

        _warmup_op(op)

        with torch.cuda.graph(g, check_input_liveness=True):
            result = torch._foreach_add(xs, ys)

        del ys[1]
        _sync_and_gc_collect()

        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_empty_tensor_not_tracked(self):
        x = torch.randn(100, device="cuda")
        empty = torch.empty(0, device="cuda")
        g = torch.cuda.CUDAGraph()

        _warmup_op(lambda: x * 2)

        with torch.cuda.graph(g, check_input_liveness=True):
            y = x * 2
            z = empty.view(0)

        self.assertEqual(empty.data_ptr(), 0)
        self.assertNotIn(empty.data_ptr(), g._external_inputs)

        del empty
        _sync_and_gc_collect()

        g.replay()
        torch.cuda.synchronize()

    def test_non_capturing_stream_inputs_not_tracked(self):
        capture_stream = torch.cuda.Stream()
        other_stream = torch.cuda.Stream()

        x = torch.randn(100, device="cuda")
        y = torch.randn(100, device="cuda")

        with torch.cuda.stream(capture_stream):
            _warmup_op(lambda: x * 2)

        g = torch.cuda.CUDAGraph()

        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(
                g, capture_error_mode="relaxed", check_input_liveness=True
            ):
                result = x * 2
                with torch.cuda.stream(other_stream):
                    other_result = y * 3

        self.assertIn(x.data_ptr(), g._external_inputs)
        self.assertNotIn(y.data_ptr(), g._external_inputs)

        del y
        _sync_and_gc_collect()

        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(result, x * 2)

    def test_shared_pool_tensor_not_tracked(self):
        pool = torch.cuda.graph_pool_handle()

        x = torch.randn(100, device="cuda")
        g1 = torch.cuda.CUDAGraph()

        _warmup_op(lambda: x + 1)

        with torch.cuda.graph(g1, pool=pool):
            y = x + 1
            w = torch.randn(100, device="cuda")

        external_input = torch.randn(100, device="cuda")
        g2 = torch.cuda.CUDAGraph()

        _warmup_op(lambda: y * 2 + external_input)

        with torch.cuda.graph(g2, pool=pool, check_input_liveness=True):
            z = y * 2 + external_input
            p = w + 1

        del y
        _sync_and_gc_collect()

        g1.replay()
        g2.replay()
        torch.cuda.synchronize()
        self.assertEqual(z, (x + 1) * 2 + external_input)

        del w
        g1.replay()
        g2.replay()
        torch.cuda.synchronize()

    def test_pinned_memory_tensor_tracked(self):
        pinned = torch.randn(100).pin_memory()
        cuda_dest = torch.empty(100, device="cuda")
        g = torch.cuda.CUDAGraph()

        def op():
            cuda_dest.copy_(pinned, non_blocking=True)
            return cuda_dest * 2

        _warmup_op(op)

        with torch.cuda.graph(g, check_input_liveness=True):
            cuda_dest.copy_(pinned, non_blocking=True)
            y = cuda_dest * 2

        self.assertIn(pinned.data_ptr(), g._external_inputs)
        self.assertIn(cuda_dest.data_ptr(), g._external_inputs)

        # This should be OK
        g.replay()
        torch.cuda.synchronize()

        del pinned
        _sync_and_gc_collect()

        # This should raise
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()


class TestCUDAGraphDebugBacktraces(TestCase):
    def test_error_message_contains_all_tracebacks(self):
        a = torch.randn(100, device="cuda")
        b = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()

        _warmup_op(lambda: a + b)

        with torch.cuda.graph(g, check_input_liveness=True):
            y = a + b

        def delete_tensors():
            nonlocal a, b
            del a, b

        delete_tensors()
        _sync_and_gc_collect()

        with self.assertRaises(RuntimeError) as ctx:
            g.replay()
        error_msg = str(ctx.exception).lower()
        self.assertIn("2 dead tensor", error_msg)
        self.assertIn("creation traceback (python)", error_msg)
        self.assertIn("creation traceback (c++)", error_msg)
        self.assertIn("deletion traceback (python)", error_msg)
        self.assertIn("delete_tensors", error_msg)
        self.assertIn("replay traceback (python)", error_msg)


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

        _warmup_op(lambda: ScaleOp.apply(x, scale))

        with torch.no_grad(), torch.cuda.graph(g, check_input_liveness=True):
            y = ScaleOp.apply(x, scale)

        del scale
        _sync_and_gc_collect()

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
        g = torch.cuda.CUDAGraph()

        _warmup_op(lambda: scale_op(x, scale))

        with torch.cuda.graph(g, check_input_liveness=True):
            y = scale_op(x, scale)

        del scale
        _sync_and_gc_collect()

        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()


if __name__ == "__main__":
    run_tests()

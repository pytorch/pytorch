# Owner(s): ["module: cuda graphs"]

import gc
import threading
import weakref
from unittest.mock import Mock, patch

import torch
from torch.cuda import (
    breakable_graph,
    breakable_graphs as bcg,
    BreakableCUDAGraph,
    force_no_graph,
    is_in_breakable_graph,
    no_graph,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestBreakableGraphAPI(TestCase):
    def test_exports(self):
        for name in bcg.__all__:
            self.assertIn(name, torch.cuda.__all__)
            self.assertIs(getattr(torch.cuda, name), getattr(bcg, name))

    def test_no_graph_outside_capture(self):
        @no_graph
        def identity(x):
            return x

        x = torch.ones(2)
        self.assertIs(identity(x), x)
        self.assertEqual(identity.__name__, "identity")
        self.assertFalse(is_in_breakable_graph())
        force_no_graph()

    def test_disabled_decorator(self):
        def fn(x):
            return x

        self.assertIs(no_graph(enable=False)(fn), fn)

    def test_capture_state_is_thread_local(self):
        ctx = breakable_graph(BreakableCUDAGraph())
        token = bcg._current_breakable_graph_ctx.set(ctx)
        states = []
        try:
            active = is_in_breakable_graph
            thread = threading.Thread(target=lambda: states.append(active()))
            thread.start()
            thread.join()
            self.assertTrue(is_in_breakable_graph())
            self.assertEqual(states, [False])
        finally:
            bcg._current_breakable_graph_ctx.reset(token)

    @parametrize("option", ["capture_stub", "barrier_fn"])
    def test_invalid_callback(self, option):
        with self.assertRaisesRegex(TypeError, f"{option}.*must be callable"):
            if option == "capture_stub":
                no_graph(capture_stub=1)
            else:
                breakable_graph(BreakableCUDAGraph(), barrier_fn=1)

    def test_lazy_pool_and_reset(self):
        graph = BreakableCUDAGraph()
        graph.replay()
        graph.reset()
        self.assertIsNone(graph._pool)
        pool = object()
        with patch.object(torch.cuda, "graph_pool_handle", return_value=pool) as create:
            self.assertIs(graph.pool(), pool)
            graph.reset()
            self.assertIs(graph.pool(), pool)
        create.assert_called_once_with()

    @parametrize("unjoined", [False, True])
    def test_capture_error_attribution(self, unjoined):
        ctx = breakable_graph(BreakableCUDAGraph())
        msg = "cudaErrorStreamCaptureUnjoined" if unjoined else "original error"
        ctx._graph_ctx = Mock()
        ctx._graph_ctx.__exit__ = Mock(side_effect=RuntimeError(msg))
        expected = "side stream was not joined" if unjoined else msg
        with self.assertRaisesRegex(RuntimeError, expected):
            ctx._end_segment()
        self.assertIsNone(ctx._graph_ctx)

    def test_initial_capture_failure(self):
        ctx = breakable_graph(BreakableCUDAGraph(pool=object()))
        graph_ctx = Mock()
        graph_ctx.__enter__ = Mock(side_effect=RuntimeError("failed entry"))
        with (
            patch.object(torch.cuda, "CUDAGraph"),
            patch.object(torch.cuda, "graph", return_value=graph_ctx),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed entry"):
                with ctx:
                    pass
        self.assertIsNone(ctx._graph_ctx)
        self.assertFalse(is_in_breakable_graph())

    def test_debug_fork_tracking(self):
        ctx = breakable_graph(BreakableCUDAGraph())
        ctx._capturing_stream_id = 1
        token = bcg._current_breakable_graph_ctx.set(ctx)
        try:
            with patch.object(bcg, "_DEBUG", True):
                bcg._on_event_record(10, 1)
                bcg._on_event_wait(10, 2)
                bcg._on_event_wait(10, 3)
                self.assertEqual(ctx._forked_stream_ids, {2, 3})
                bcg._on_event_record(11, 2)
                bcg._on_event_wait(11, 1)
                self.assertEqual(ctx._forked_stream_ids, {3})
                bcg._on_event_record(12, 1)
                bcg._on_event_wait(12, 2)
                self.assertEqual(ctx._forked_stream_ids, {2, 3})
        finally:
            bcg._current_breakable_graph_ctx.reset(token)


class TestBreakableCUDAGraph(TestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(torch.compiler.config.patch(force_cudagraph_gc=True))

    def warmup(self, fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)

    def test_no_breaks(self, device):
        x = torch.ones(8, device=device)
        out = torch.empty_like(x)

        def fn():
            out.copy_(x * 2 + 1)

        self.warmup(fn)
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            fn()
        self.assertEqual(len(graph._segments), 1)
        self.assertIsInstance(graph._segments[0], torch.cuda.CUDAGraph)
        x.fill_(3)
        graph.replay()
        self.assertEqual(out, x * 2 + 1)

    @parametrize("position", [0, 1, 2])
    @parametrize("enable", [False, True])
    def test_eager_break_placement(self, device, position, enable):
        x = torch.ones(8, device=device)
        out = torch.empty_like(x)

        @no_graph(enable=enable)
        def eager():
            out.add_(3)

        def fn():
            out.copy_(x)
            for i in range(3):
                if i == position:
                    eager()
                else:
                    out.mul_(2)

        self.warmup(fn)
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            fn()
        self.assertEqual(len(graph._segments), 3 if enable else 1)
        x.fill_(5)
        graph.replay()
        expected = torch.full_like(x, 5)
        for i in range(3):
            if i == position:
                expected.add_(3)
            else:
                expected.mul_(2)
        self.assertEqual(out, expected)

    def test_force_no_graph(self, device):
        x = torch.ones(8, device=device)

        def fn():
            x.mul_(2)
            force_no_graph()
            x.add_(1)
            force_no_graph()
            x.mul_(3)

        self.warmup(fn)
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            fn()
        self.assertEqual(len(graph._segments), 5)
        x.fill_(2)
        graph.replay()
        self.assertEqual(x, torch.full_like(x, 15))

    def test_capture_stub_and_barrier(self, device):
        x = torch.ones(8, device=device)
        calls = []

        def stub(buf):
            calls.append("stub")
            buf.zero_()

        @no_graph(capture_stub=stub)
        def eager(buf):
            calls.append("eager")
            buf.mul_(3)

        def fn():
            x.add_(1)
            eager(x)
            x.add_(2)

        self.warmup(fn)
        calls.clear()
        graph = BreakableCUDAGraph()
        with breakable_graph(graph, barrier_fn=lambda: calls.append("barrier")):
            fn()
        self.assertEqual(calls, ["barrier", "stub"])
        calls.clear()
        x.fill_(2)
        graph.replay()
        self.assertEqual(calls, ["eager"])
        self.assertEqual(x, torch.full_like(x, 11))

    @parametrize("container", ["tensor", "tuple", "list", "dict"])
    def test_cuda_returns_rejected(self, device, container):
        x = torch.ones(8, device=device)

        @no_graph
        def eager(buf):
            result = buf * 2
            if container == "tuple":
                return (result,)
            if container == "list":
                return [result]
            if container == "dict":
                return {"x": [result]}
            return result

        self.warmup(lambda: eager(x))
        graph = BreakableCUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "returns one or more CUDA tensors"):
            with breakable_graph(graph):
                eager(x)
        self.assertFalse(is_in_breakable_graph())
        self.assertFalse(torch.cuda.is_current_stream_capturing())
        graph.reset()

    def test_nested_args_and_view_metadata(self, device):
        base = torch.arange(48, device=device, dtype=torch.float32).reshape(6, 8)
        x = base[1::2, 1::2].t()
        out = torch.empty_like(x)
        cpu_arg = torch.ones(1)
        metadata = []

        @no_graph
        def eager(args, *, outputs, scale):
            src = args["input"][0]
            metadata.append((src.shape, src.stride(), src.storage_offset()))
            outputs[0].copy_(src * scale)
            self.assertIs(args["cpu"], cpu_arg)
            return {"scalar": 3, "cpu": cpu_arg}

        def fn():
            result = eager({"input": [x], "cpu": cpu_arg}, outputs=(out,), scale=2)
            out.add_(result["scalar"])

        self.warmup(fn)
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            fn()
        base.add_(10)
        graph.replay()
        self.assertEqual(out, x * 2 + 3)
        self.assertEqual(metadata[-1], (x.shape, x.stride(), x.storage_offset()))

    @parametrize("conjugate", [False, True])
    @parametrize("negative", [False, True])
    def test_lazy_view_bits(self, device, conjugate, negative):
        base = torch.tensor([1 + 2j, 3 + 4j], device=device)
        x = base.conj() if conjugate else base
        x = x._neg_view() if negative else x
        out = torch.empty_like(x)

        @no_graph
        def eager(src, dst):
            dst.copy_(src)

        self.warmup(lambda: eager(x, out))
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            eager(x, out)
        base.add_(1 + 1j)
        graph.replay()
        self.assertEqual(out, x)

    def test_nested_decorators_and_capture_state(self, device):
        x = torch.ones(8, device=device)
        states = []

        @no_graph
        def inner(buf):
            states.append(is_in_breakable_graph())
            buf.mul_(2)

        @no_graph
        def outer(buf):
            inner(buf)
            buf.add_(1)

        self.warmup(lambda: outer(x))
        states.clear()
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            self.assertTrue(is_in_breakable_graph())
            outer(x)
        self.assertEqual(states, [True])
        self.assertEqual(len(graph._segments), 3)
        self.assertFalse(is_in_breakable_graph())
        states.clear()
        x.fill_(3)
        graph.replay()
        self.assertEqual(states, [False])
        self.assertEqual(x, torch.full_like(x, 7))

    def test_reset_recapture_and_pool_sharing(self, device):
        x = torch.ones(8, device=device)
        self.warmup(lambda: x.add_(1))
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            x.add_(1)
            force_no_graph()
            x.mul_(2)
        pool = graph.pool()
        segments = [g for g in graph._segments if isinstance(g, torch.cuda.CUDAGraph)]
        self.assertEqual([g.pool() for g in segments], [pool, pool])
        graph.reset()
        self.assertEqual(graph._segments, [])
        self.assertEqual(graph.pool(), pool)
        with breakable_graph(graph):
            x.add_(3)
        other = BreakableCUDAGraph(pool=graph.pool())
        with breakable_graph(other):
            x.mul_(2)
        x.fill_(1)
        graph.replay()
        other.replay()
        self.assertEqual(x, torch.full_like(x, 8))

    def test_mem_pool(self, device):
        pool = torch.cuda.MemPool()
        x = torch.ones(8, device=device)
        self.warmup(lambda: x.add_(1))
        graph = BreakableCUDAGraph(pool=pool)
        with breakable_graph(graph):
            x.add_(1)
            force_no_graph()
            x.mul_(2)
        self.assertIs(graph.pool(), pool)
        x.fill_(2)
        graph.replay()
        self.assertEqual(x, torch.full_like(x, 6))

    @parametrize("eager_failure", [False, True])
    def test_capture_exception_cleanup(self, device, eager_failure):
        x = torch.ones(8, device=device)
        self.warmup(lambda: x.add_(1))
        ambient = torch.cuda.current_stream()

        @no_graph
        def failing():
            raise RuntimeError("intentional failure")

        graph = BreakableCUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with breakable_graph(graph):
                x.add_(1)
                if eager_failure:
                    failing()
                else:
                    raise RuntimeError("intentional failure")
        self.assertFalse(is_in_breakable_graph())
        self.assertFalse(torch.cuda.is_current_stream_capturing())
        self.assertEqual(torch.cuda.current_stream(), ambient)
        graph.reset()
        with breakable_graph(graph):
            x.add_(2)
        x.zero_()
        graph.replay()
        self.assertEqual(x, torch.full_like(x, 2))

    def test_nested_captures_rejected(self, device):
        x = torch.ones(8, device=device)
        self.warmup(lambda: x.add_(1))
        graph = BreakableCUDAGraph()
        with breakable_graph(graph):
            x.add_(1)
            with self.assertRaisesRegex(RuntimeError, "nested breakable_graph"):
                with breakable_graph(BreakableCUDAGraph()):
                    pass
        x.zero_()
        graph.replay()
        self.assertEqual(x, torch.ones_like(x))

    def test_joined_side_stream(self, device):
        capture, side = torch.cuda.Stream(), torch.cuda.Stream()
        self.assertNotEqual(capture.cuda_stream, side.cuda_stream)
        x = torch.ones(8, device=device)

        @no_graph
        def eager(buf):
            buf.mul_(3)

        def fn():
            x.add_(1)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                x.add_(2)
            torch.cuda.current_stream().wait_stream(side)
            eager(x)
            x.add_(4)

        self.warmup(fn)
        graph = BreakableCUDAGraph()
        with breakable_graph(graph, stream=capture):
            fn()
        x.fill_(2)
        graph.replay()
        self.assertEqual(x, torch.full_like(x, 19))

    @parametrize("at_exit", [False, True])
    @parametrize("debug", [False, True])
    def test_unjoined_side_stream(self, device, at_exit, debug):
        capture, side = torch.cuda.Stream(), torch.cuda.Stream()
        self.assertNotEqual(capture.cuda_stream, side.cuda_stream)
        x = torch.ones(8, device=device)
        self.warmup(lambda: x.add_(1))
        ambient = torch.cuda.current_stream()
        graph = BreakableCUDAGraph()
        expected = "side stream was not joined"
        if debug:
            expected += f".*Unjoined side-stream id\\(s\\): \\[{side.cuda_stream}\\]"
        with patch.object(bcg, "_DEBUG", debug):
            with self.assertRaisesRegex(RuntimeError, expected):
                with breakable_graph(graph, stream=capture):
                    x.add_(1)
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        x.add_(2)
                    if not at_exit:
                        force_no_graph()
        self.assertFalse(is_in_breakable_graph())
        self.assertEqual(torch.cuda.current_stream(), ambient)
        graph.reset()
        with breakable_graph(graph, stream=capture):
            x.add_(3)
        x.zero_()
        graph.replay()
        self.assertEqual(x, torch.full_like(x, 3))

    def test_eager_arguments_do_not_retain_storage(self, device):
        @no_graph
        def consume(buf):
            buf.add_(1)

        graph = BreakableCUDAGraph()
        x = torch.empty(4099, device=device)
        ptr = x.data_ptr()
        self.warmup(lambda buf=x: consume(buf))
        with breakable_graph(graph):
            consume(x)
        tensor_ref = weakref.ref(x)
        storage_ref = weakref.ref(x.untyped_storage())
        torch.cuda.synchronize()
        del x
        gc.collect()
        self.assertIsNone(tensor_ref())
        self.assertIsNone(storage_ref())
        replacement = torch.empty(4099, device=device)
        self.assertEqual(replacement.data_ptr(), ptr)

    def test_forward_backward_with_eager_break(self, device):
        x = torch.ones(8, device=device, requires_grad=True)
        grad = torch.ones_like(x)

        @no_graph
        def eager(buf):
            buf.mul_(2)

        def fn():
            x.grad = None
            out = x * 3
            eager(out)
            out.backward(grad)

        with torch.autograd.set_multithreading_enabled(False):
            self.warmup(fn)
            graph = BreakableCUDAGraph()
            with breakable_graph(graph):
                fn()
            grad.fill_(2)
            graph.replay()
        self.assertEqual(x.grad, torch.full_like(x, 12))


instantiate_parametrized_tests(TestBreakableGraphAPI)
instantiate_device_type_tests(TestBreakableCUDAGraph, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

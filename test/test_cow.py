# Owner(s): ["module: cuda"]

import contextlib
import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    TEST_CUDA,
    TEST_CUDA_GRAPH,
    TestCase,
)
from torch.utils._triton import has_triton


TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)


@unittest.skipIf(TEST_CUDAMALLOCASYNC, "requires native allocator stream metadata")
class TestCOW(TestCase):
    @parametrize("captured", [False, True])
    def test_readonly_sharing(self, device, captured):
        a = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            with (
                torch.cuda.graph(graph, stream=a)
                if captured
                else contextlib.nullcontext()
            ):
                x = torch.ones(1024, device=device)
                y = x._lazy_clone()
                z = y._lazy_clone()
                out = z * 2
            if captured:
                graph.replay()
            self.assertEqual(x.const_data_ptr(), y.const_data_ptr())
            self.assertEqual(x.const_data_ptr(), z.const_data_ptr())
            self.assertTrue(torch._C._is_cow_tensor(y))
            self.assertTrue(torch._C._is_cow_tensor(z))
            self.assertEqual(out, torch.full_like(out, 2))
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    @parametrize("pointer", [False, True])
    def test_materialize_on_current_stream(self, device, captured, pointer):
        a = torch.cuda.Stream(device=device)
        b = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            inp = torch.ones(1024, device=device)
            with (
                torch.cuda.graph(graph, stream=a)
                if captured
                else contextlib.nullcontext()
            ):
                x = inp + 0
                y = x._lazy_clone()
                b.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(10_000_000)
                    if pointer:
                        y.data_ptr()
                    else:
                        y.add_(1)
                    out = y * 2
                x.add_(2)
                a.wait_stream(b)
            self.assertNotEqual(x.const_data_ptr(), y.const_data_ptr())
            self.assertFalse(torch._C._is_cow_tensor(y))
            for value in [3, 7] if captured else [1]:
                if captured:
                    inp.fill_(value)
                    graph.replay()
                self.assertEqual(x, inp + 2)
                self.assertEqual(y, inp + (0 if pointer else 1))
                self.assertEqual(out, (inp + (0 if pointer else 1)) * 2)
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    @parametrize("lazy_read", [False, True])
    def test_steal_waits_for_readers(self, device, captured, lazy_read):
        a, b, c = [torch.cuda.Stream(device=device) for _ in range(3)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            inp = torch.ones(1024, device=device)
            with (
                torch.cuda.graph(graph, stream=a)
                if captured
                else contextlib.nullcontext()
            ):
                x = inp + 0
                y = x._lazy_clone()
                ptr = x.const_data_ptr()
                b.wait_stream(a)
                c.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(10_000_000)
                    out = x._lazy_clone() if lazy_read else x * 2
                a.wait_stream(b)
                del x
                with torch.cuda.stream(c):
                    y.add_(1)
                a.wait_stream(c)
            self.assertEqual(y.const_data_ptr(), ptr)
            for value in [3, 7] if captured else [1]:
                if captured:
                    inp.fill_(value)
                    graph.replay()
                self.assertEqual(out, inp if lazy_read else inp * 2)
                self.assertEqual(y, inp + 1)
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    @parametrize("read_before_clone", [False, True])
    def test_pointer_materialization_preserves_older_reads(
        self, device, captured, read_before_clone
    ):
        a = torch.cuda.Stream(device=device)
        b = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            inp = torch.ones(1024, device=device)
            with (
                torch.cuda.graph(graph, stream=a)
                if captured
                else contextlib.nullcontext()
            ):
                x = inp + 0
                if not read_before_clone:
                    y = x._lazy_clone()
                b.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(10_000_000)
                    out = x * 2
                if read_before_clone:
                    y = x._lazy_clone()
                x.data_ptr()
                y.add_(1)
                a.wait_stream(b)
            for value in [3, 7] if captured else [1]:
                if captured:
                    inp.fill_(value)
                    graph.replay()
                self.assertEqual(out, inp * 2)
                self.assertEqual(x, inp)
                self.assertEqual(y, inp + 1)
        torch.cuda.current_stream(device).wait_stream(a)

    def test_replacement_keeps_allocation_stream(self, device):
        a = torch.cuda.Stream(device=device)
        b = torch.cuda.Stream(device=device)
        with torch.cuda.stream(a):
            x = torch.ones(1024, device=device)
            y = x._lazy_clone()
            b.wait_stream(a)
            with torch.cuda.stream(b):
                y.add_(1)
            a.wait_stream(b)
            torch.cuda._sleep(10_000_000)
            out = y * 2
            ptr = y.const_data_ptr()
            del y
            with torch.cuda.stream(b):
                scratch = torch.empty(1024, device=device)
                self.assertNotEqual(scratch.data_ptr(), ptr)
                scratch.fill_(99)
            a.wait_stream(b)
            self.assertEqual(out, torch.full_like(out, 4))
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    def test_pointer_materialization_orders_reuse(self, device, captured):
        a = torch.cuda.Stream(device=device)
        b = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            with (
                torch.cuda.graph(graph, stream=a)
                if captured
                else contextlib.nullcontext()
            ):
                x = torch.ones(1024, device=device)
                y = x._lazy_clone()
                b.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(10_000_000)
                    ptr = y.data_ptr()
                del y
                replacement = torch.empty_like(x)
                self.assertEqual(replacement.data_ptr(), ptr)
                replacement.fill_(99)
                a.wait_stream(b)
            if captured:
                graph.replay()
            self.assertEqual(replacement, torch.full_like(replacement, 99))
            self.assertEqual(x, torch.ones_like(x))
        torch.cuda.current_stream(device).wait_stream(a)

    def test_retired_allocation_is_bounded_and_released(self, device):
        initial = torch.cuda.memory_allocated(device)
        x = torch.ones(16 * 1024, device=device)
        nbytes = x.nbytes
        for _ in range(4):
            y = x._lazy_clone()
            x.add_(1)
            del y
            self.assertEqual(torch.cuda.memory_allocated(device), initial + 2 * nbytes)
        weak_storage = x.untyped_storage()._weak_ref()
        try:
            del x
            self.assertTrue(torch.UntypedStorage._expired(weak_storage))
            self.assertEqual(torch.cuda.memory_allocated(device), initial)
        finally:
            torch.UntypedStorage._free_weak_ref(weak_storage)

    def test_graph_input_stays_at_its_address(self, device):
        a = torch.cuda.Stream(device=device)
        inp = torch.ones(1024, device=device)
        ptr = inp.data_ptr()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=a):
            y = inp._lazy_clone()
            out = y * 2
        del y
        for value in [3, 7]:
            inp.fill_(value)
            graph.replay()
            self.assertEqual(out, inp * 2)
            self.assertEqual(inp.data_ptr(), ptr)
        snapshot = inp._lazy_clone()
        self.assertFalse(torch._C._is_cow_tensor(snapshot))
        inp.add_(1)
        graph.replay()
        self.assertEqual(out, inp * 2)
        self.assertEqual(snapshot, inp - 1)

    def test_capture_boundary_errors(self, device):
        a = torch.cuda.Stream(device=device)
        x = torch.ones(1024, device=device)
        y = x._lazy_clone()
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "Materialize COW graph inputs"):
            with torch.cuda.graph(graph, stream=a):
                y * 2
        self.assertTrue(torch._C._is_cow_tensor(y))
        del y
        x.add_(1)

        inp = torch.ones(1024, device=device)
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "Cannot relocate a COW graph input"):
            with torch.cuda.graph(graph, stream=a):
                y = inp._lazy_clone()
                y.add_(1)
        del y
        inp.add_(1)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=a):
            x = torch.ones(1024, device=device)
            y = x._lazy_clone()
        with self.assertRaisesRegex(
            RuntimeError, "across CUDA graph capture boundaries"
        ):
            y.data_ptr()
        del x
        graph.replay()
        y.add_(1)
        self.assertEqual(y, torch.full_like(y, 2))
        graph.replay()
        self.assertEqual(y, torch.ones_like(y))

    def test_graph_output_snapshot(self, device):
        a = torch.cuda.Stream(device=device)
        with torch.cuda.stream(a):
            inp = torch.ones(1024, device=device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=a):
                out = inp * 2
            graph.replay()
            snapshot = out._lazy_clone()
            self.assertFalse(torch._C._is_cow_tensor(snapshot))
            inp.fill_(3)
            graph.replay()
            self.assertEqual(snapshot, torch.full_like(snapshot, 2))
            self.assertEqual(out, torch.full_like(out, 6))
        torch.cuda.current_stream(device).wait_stream(a)

    @unittest.skipUnless(has_triton(), "requires Triton")
    def test_compile_reduce_overhead(self, device):
        def fn(x):
            y = x + 1
            z = y._lazy_clone()
            return y.sin() + z.cos()

        compiled = torch.compile(fn, fullgraph=True, mode="reduce-overhead")
        x = torch.ones(1024, device=device)
        for value in [1, 3, 7, 11]:
            torch.compiler.cudagraph_mark_step_begin()
            x.fill_(value)
            self.assertEqual(compiled(x), fn(x))


class TestCOWFallback(TestCase):
    def test_clone_on_another_stream(self, device):
        a = torch.cuda.current_stream(device)
        b = torch.cuda.Stream(device=device)
        x = torch.ones(1024, device=device)
        b.wait_stream(a)
        with torch.cuda.stream(b):
            y = x._lazy_clone()
            self.assertFalse(torch._C._is_cow_tensor(y))
            y.add_(1)
        a.wait_stream(b)
        self.assertEqual(x, torch.ones_like(x))
        self.assertEqual(y, torch.full_like(y, 2))

    @unittest.skipUnless(TEST_CUDAMALLOCASYNC, "requires an allocator without metadata")
    def test_unknown_allocator_uses_eager_copy(self, device):
        x = torch.ones(1024, device=device)
        y = x._lazy_clone()
        self.assertFalse(torch._C._is_cow_tensor(x))
        self.assertFalse(torch._C._is_cow_tensor(y))
        x.add_(1)
        self.assertEqual(y, torch.ones_like(y))


instantiate_device_type_tests(
    TestCOW, globals(), only_for="cuda" if TEST_CUDA_GRAPH else []
)
instantiate_device_type_tests(TestCOWFallback, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

# Owner(s): ["module: cuda"]

import contextlib
import unittest

import torch
from torch.cuda._utils import _check_cuda_bindings, _cuda_bindings_runtime
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    requires_cuda_python_bindings,
    run_tests,
    skipIfRocm,
    TEST_CUDA,
    TEST_CUDA_GRAPH,
    TestCase,
)
from torch.utils._triton import has_triton


TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)


def _capture_ancestors(stream):
    info = _check_cuda_bindings(
        _cuda_bindings_runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
    )
    pending = [int(node) for node in info[3]]
    ancestors = set()
    while pending:
        node = pending.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        count = _check_cuda_bindings(
            _cuda_bindings_runtime.cudaGraphNodeGetDependencies(node)
        )[-1]
        if count:
            parents = _check_cuda_bindings(
                _cuda_bindings_runtime.cudaGraphNodeGetDependencies(node, count)
            )[0]
            pending.extend(int(parent) for parent in parents)
    return ancestors


@unittest.skipUnless(TEST_CUDA_GRAPH, "requires CUDA graph support")
@unittest.skipIf(TEST_CUDAMALLOCASYNC, "requires native allocator stream metadata")
class TestCOW(TestCase):
    hw_classification = HardwareClassification.CUDA

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
    def test_materialize_on_another_stream(self, device, captured, pointer):
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

    @skipIfRocm
    @requires_cuda_python_bindings
    def test_materialization_does_not_wait_for_operation_stream(self, device):
        a = torch.cuda.Stream(device=device)
        b = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            with torch.cuda.graph(graph, stream=a):
                x = torch.ones(1024, device=device)
                y = x._lazy_clone()
                b.wait_stream(a)
                with torch.cuda.stream(b):
                    before = _capture_ancestors(b)
                    late = x + 2
                    late_nodes = _capture_ancestors(b) - before
                    y.data_ptr()
                self.assertTrue(late_nodes)
                self.assertTrue(_capture_ancestors(a).isdisjoint(late_nodes))
                a.wait_stream(b)
            graph.replay()
            self.assertEqual(x, torch.ones_like(x))
            self.assertEqual(y, x)
            self.assertEqual(late, torch.full_like(late, 3))
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("resize_to_zero", [False, True])
    def test_resize_cow_storage(self, device, resize_to_zero):
        x = torch.ones(8, device=device)
        y = x._lazy_clone()
        storage = y.untyped_storage()
        if resize_to_zero:
            storage.resize_(0)
        storage.resize_(64)
        if resize_to_zero:
            y.fill_(1)
        self.assertFalse(torch._C._is_cow_tensor(y))
        self.assertNotEqual(y.const_data_ptr(), x.const_data_ptr())
        self.assertEqual(y.sum(), 8)
        storage.resize_(128)
        z = y._lazy_clone()
        y.add_(1)
        self.assertEqual(x, torch.ones_like(x))
        self.assertEqual(z, x)
        self.assertEqual(y, torch.full_like(y, 2))

    def test_is_cow_tensor_during_capture(self, device):
        x = torch.ones(8, device=device)
        y = x._lazy_clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.assertTrue(torch._C._is_cow_tensor(x))
            self.assertTrue(torch._C._is_cow_tensor(y))
            out = torch.ones(8, device=device)
        graph.replay()
        self.assertEqual(out, x)

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

    @parametrize("complete", [False, True])
    def test_retired_allocation_is_bounded_and_released(self, device, complete):
        initial = torch.cuda.memory_allocated(device)
        x = torch.ones(16 * 1024, device=device)
        nbytes = x.nbytes
        for _ in range(4):
            y = x._lazy_clone()
            x.add_(1)
            if complete:
                torch._C._cow_record_done(x)
            del y
            self.assertEqual(
                torch.cuda.memory_allocated(device),
                initial + (1 if complete else 2) * nbytes,
            )
        weak_storage = x.untyped_storage()._weak_ref()
        try:
            del x
            self.assertTrue(torch.UntypedStorage._expired(weak_storage))
            self.assertEqual(torch.cuda.memory_allocated(device), initial)
        finally:
            torch.UntypedStorage._free_weak_ref(weak_storage)

    @parametrize("captured", [False, True])
    @parametrize("pointer", [False, True])
    def test_completed_materialization_allows_stealing(self, device, captured, pointer):
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
                ptr = y.const_data_ptr()
                b.wait_stream(a)
                c.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(10_000_000)
                    if pointer:
                        x.data_ptr()
                    else:
                        x.add_(2)
                    torch._C._cow_record_done(x)
                with torch.cuda.stream(c):
                    y.add_(1)
                a.wait_stream(c)
            self.assertEqual(y.const_data_ptr(), ptr)
            self.assertNotEqual(x.const_data_ptr(), ptr)
            for value in [3, 7] if captured else [1]:
                if captured:
                    inp.fill_(value)
                    graph.replay()
                self.assertEqual(x, inp + (0 if pointer else 2))
                self.assertEqual(y, inp + 1)
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    @parametrize("steal", [False, True])
    def test_all_completion_events_order_stealing_and_reuse(
        self, device, captured, steal
    ):
        a, b, c, d = [torch.cuda.Stream(device=device) for _ in range(4)]
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
                z = y._lazy_clone()
                ptr = y.const_data_ptr()
                for stream in [b, c, d]:
                    stream.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(20_000_000)
                    first = x * 2
                    torch._C._cow_record_done(x)
                del x
                with torch.cuda.stream(c):
                    second = z * 3
                    torch._C._cow_record_done(z)
                    torch._C._cow_record_done(z)
                del z
                if steal:
                    with torch.cuda.stream(d):
                        y.add_(1)
                    a.wait_stream(d)
                else:
                    torch._C._cow_record_done(y)
                    del y
                    y = torch.empty(1024, device=device)
                    y.fill_(99)
                self.assertEqual(y.const_data_ptr(), ptr)
            for value in [3, 7] if captured else [1]:
                if captured:
                    inp.fill_(value)
                    graph.replay()
                self.assertEqual(first, inp * 2)
                self.assertEqual(second, inp * 3)
                self.assertEqual(y, inp + 1 if steal else torch.full_like(inp, 99))
        torch.cuda.current_stream(device).wait_stream(a)

    @skipIfRocm
    @requires_cuda_python_bindings
    @parametrize("materialize", [False, True])
    def test_completion_does_not_wait_for_later_work(self, device, materialize):
        a, b, c = [torch.cuda.Stream(device=device) for _ in range(3)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            with torch.cuda.graph(graph, stream=a):
                x = torch.ones(1024, device=device)
                y = x._lazy_clone()
                ptr = y.const_data_ptr()
                b.wait_stream(a)
                c.wait_stream(a)
                with torch.cuda.stream(b):
                    out = x.add_(2) if materialize else x * 3
                    torch._C._cow_record_done(x)
                    self.assertEqual(torch._C._is_cow_tensor(x), not materialize)
                del x
                before = _capture_ancestors(a)
                late = torch.ones(1024, device=device)
                late_nodes = _capture_ancestors(a) - before
                with torch.cuda.stream(c):
                    y.add_(1)
                self.assertTrue(late_nodes)
                self.assertTrue(_capture_ancestors(c).isdisjoint(late_nodes))
                self.assertEqual(y.const_data_ptr(), ptr)
                a.wait_stream(c)
            graph.replay()
            self.assertEqual(out, torch.full_like(out, 3))
            self.assertEqual(y, torch.full_like(y, 2))
            self.assertEqual(late, torch.ones_like(late))
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    def test_access_after_completion_restores_lifetime_contract(self, device, captured):
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
                ptr = y.const_data_ptr()
                torch._C._cow_record_done(x)
                b.wait_stream(a)
                c.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(10_000_000)
                    out = x * 2
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
                self.assertEqual(out, inp * 2)
                self.assertEqual(y, inp + 1)
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    def test_completed_references_allow_chained_clones(self, device, captured):
        a = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            with (
                torch.cuda.graph(graph, stream=a)
                if captured
                else contextlib.nullcontext()
            ):
                x = torch.ones(1024, device=device)
                snapshots = []
                for _ in range(3):
                    y = x._lazy_clone()
                    z = y._lazy_clone()
                    self.assertTrue(torch._C._is_cow_tensor(x))
                    ptr = x.const_data_ptr()
                    x.add_(1)
                    torch._C._cow_record_done(x.view(-1))
                    torch._C._cow_record_done(x)
                    y.add_(10)
                    # Completing x did not complete its descendants.
                    self.assertNotEqual(y.const_data_ptr(), ptr)
                    torch._C._cow_record_done(y)
                    z.add_(20)
                    self.assertEqual(z.const_data_ptr(), ptr)
                    snapshots.append((y, z))
            for _ in range(2 if captured else 1):
                if captured:
                    graph.replay()
                self.assertEqual(x, torch.full_like(x, 4))
                for i, (y, z) in enumerate(snapshots):
                    self.assertEqual(y, torch.full_like(y, i + 11))
                    self.assertEqual(z, torch.full_like(z, i + 21))
        torch.cuda.current_stream(device).wait_stream(a)

    def test_completion_capture_boundaries(self, device):
        a = torch.cuda.Stream(device=device)
        x = torch.ones(1024, device=device)
        y = x._lazy_clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=a):
            with self.assertRaisesRegex(RuntimeError, "complete COW uses across"):
                torch._C._cow_record_done(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=a):
            x = torch.ones(1024, device=device)
            y = x._lazy_clone()
            x.add_(1)
        for tensor in [x, y]:
            with self.assertRaisesRegex(RuntimeError, "complete COW uses across"):
                torch._C._cow_record_done(tensor)

    @parametrize("captured", [False, True])
    def test_completion_covers_reads_before_cow(self, device, captured):
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
                b.wait_stream(a)
                c.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(10_000_000)
                    out = x * 2
                y = x._lazy_clone()
                ptr = y.const_data_ptr()
                x.data_ptr()
                with torch.cuda.stream(b):
                    b.wait_stream(a)
                    torch._C._cow_record_done(x)
                with torch.cuda.stream(c):
                    y.add_(1)
                a.wait_stream(c)
            self.assertEqual(y.const_data_ptr(), ptr)
            for value in [3, 7] if captured else [1]:
                if captured:
                    inp.fill_(value)
                    graph.replay()
                self.assertEqual(out, inp * 2)
                self.assertEqual(x, inp)
                self.assertEqual(y, inp + 1)
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("captured", [False, True])
    def test_completion_can_be_renewed_on_another_stream(self, device, captured):
        a, b, c, d = [torch.cuda.Stream(device=device) for _ in range(4)]
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
                ptr = y.const_data_ptr()
                for stream in [b, c, d]:
                    stream.wait_stream(a)
                with torch.cuda.stream(b):
                    torch.cuda._sleep(20_000_000)
                    first = x * 2
                    torch._C._cow_record_done(x)
                with torch.cuda.stream(c):
                    second = x * 3
                    torch._C._cow_record_done(x)
                del x
                with torch.cuda.stream(d):
                    y.add_(1)
                a.wait_stream(d)
            self.assertEqual(y.const_data_ptr(), ptr)
            for value in [3, 7] if captured else [1]:
                if captured:
                    inp.fill_(value)
                    graph.replay()
                self.assertEqual(first, inp * 2)
                self.assertEqual(second, inp * 3)
                self.assertEqual(y, inp + 1)
        torch.cuda.current_stream(device).wait_stream(a)

    @parametrize("same_stream", [False, True])
    def test_lazy_clone_graph_input_rejected(self, device, same_stream):
        a = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        with torch.cuda.stream(a if same_stream else current):
            inp = torch.ones(1024, device=device)
        ptr = inp.data_ptr()
        a.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(a):
            with torch.cuda.graph(graph, stream=a):
                with self.assertRaisesRegex(
                    RuntimeError, "Cannot lazily clone a CUDA graph input"
                ):
                    inp._lazy_clone()
                readonly = inp * 2
                out = inp.clone()
                out.add_(1)
            self.assertFalse(torch._C._is_cow_tensor(inp))
            self.assertEqual(inp.data_ptr(), ptr)
            for value in [3, 7]:
                inp.fill_(value)
                graph.replay()
                self.assertEqual(readonly, inp * 2)
                self.assertEqual(out, inp + 1)
        current.wait_stream(a)

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
    hw_classification = HardwareClassification.CUDA

    @parametrize("name", ["_is_cow_tensor", "_cow_record_done"])
    def test_fake_tensor_unsupported(self, device, name):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            x = torch.ones(8, device=device)
            with self.assertRaisesRegex(RuntimeError, "FakeTensors"):
                getattr(torch._C, name)(x)

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


instantiate_device_type_tests(TestCOW, globals(), only_for="cuda")
instantiate_device_type_tests(TestCOWFallback, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

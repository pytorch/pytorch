# Owner(s): ["module: cuda"]

import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_CUDA,
    TEST_CUDA_GRAPH,
    TestCase,
)


TEST_NATIVE_ALLOCATOR = TEST_CUDA and torch.cuda.get_allocator_backend() == "native"

# Long enough for a racing access to observably run before the pending one.
SLEEP_CYCLES = 50_000_000


def is_cow(t):
    return torch._C._is_cow_tensor(t)


def address(t):
    # Unlike data_ptr(), this does not materialize.
    return torch._C._data_address(t)


release = torch._C._lazy_clone_release


# Lazy clones of CUDA tensors should be safe to use with the synchronization an
# ordinary clone made on the current stream would need.
@unittest.skipUnless(TEST_NATIVE_ALLOCATOR, "requires the native CUDA allocator")
class TestCOWCUDA(TestCase):
    def setUp(self):
        super().setUp()
        # Loading a kernel on its first launch can synchronize, hiding races.
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.cuda._sleep(1)
            t = torch.ones(16, device="cuda")
            (t * 2).add_(1).fill_(7)
            t.clone()
        torch.cuda.synchronize()

    def test_materialize_on_current_stream(self, device):
        # The copy must be ordered after x's producer, still pending on side.
        side = torch.cuda.Stream(device=device)
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            x = torch.ones(1 << 20, device=device)
            y = x._lazy_clone()
            self.assertTrue(is_cow(y))
            x.add_(1)
            self.assertFalse(is_cow(x))
            x_out, y_out = x * 1, y * 1
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(x_out, torch.full_like(x_out, 2))
        self.assertEqual(y_out, torch.ones_like(y_out))

    def test_last_reference_copies(self, device):
        x = torch.ones(1 << 20, device=device)
        ptr = x.data_ptr()
        y = x._lazy_clone()
        del x
        y.add_(1)
        self.assertFalse(is_cow(y))
        self.assertNotEqual(y.data_ptr(), ptr)
        self.assertEqual(y, torch.full_like(y, 2))

    def test_no_steal_into_pending_read(self, device):
        # A released reference's read, made safe for its release with
        # record_stream(), must not race with a write to another reference.
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            out = x * 2
        x.record_stream(side)
        del x
        y.fill_(7)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(out, torch.full_like(out, 2))
        self.assertEqual(y, torch.full_like(y, 7))

    def test_materialized_keeps_data_for_pending_read(self, device):
        # x's read is pending on another stream; materializing x and releasing
        # the other reference must not free the memory it reads.
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            out = x * 2
        x.data_ptr()
        del y
        scratch = torch.empty_like(x)
        scratch.fill_(99)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(out, torch.full_like(out, 2))

    def test_record_stream(self, device):
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            out = x * 2
        x.record_stream(side)
        del x, y
        scratch = torch.empty(1 << 20, device=device)
        scratch.fill_(99)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(out, torch.full_like(out, 2))

    def test_record_stream_after_materialize(self, device):
        # The read of x on side was enqueued before x was materialized, so
        # it reads the data x was materialized from.
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            out = x * 2
        x.add_(1)
        x.record_stream(side)
        del x, y
        scratch = torch.empty(1 << 20, device=device)
        scratch.fill_(99)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(out, torch.full_like(out, 2))

    def test_materialized_keeps_data_until_released(self, device):
        x = torch.ones(1 << 20, device=device)
        nbytes = x.nbytes
        base = torch.cuda.memory_allocated(device)
        y = x._lazy_clone()
        y.add_(1)
        self.assertEqual(torch.cuda.memory_allocated(device), base + nbytes)
        del x
        self.assertEqual(torch.cuda.memory_allocated(device), base + nbytes)
        del y
        self.assertEqual(torch.cuda.memory_allocated(device), base - nbytes)

    def test_lazy_clone_of_materialized_copies(self, device):
        x = torch.ones(16, device=device)
        y = x._lazy_clone()
        x.add_(1)
        z = x._lazy_clone()
        self.assertFalse(is_cow(x))
        self.assertFalse(is_cow(z))
        self.assertNotEqual(z.data_ptr(), x.data_ptr())
        self.assertEqual(z, torch.full_like(z, 2))
        self.assertEqual(y, torch.ones_like(y))

    def test_lazy_clone_on_other_stream_copies(self, device):
        side = torch.cuda.Stream(device=device)
        x = torch.ones(16, device=device)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            z = x._lazy_clone()
            w = y._lazy_clone()
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertFalse(is_cow(z))
        self.assertFalse(is_cow(w))
        self.assertEqual(z, x)
        self.assertEqual(w, x)

    def test_release_allows_steal(self, device):
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        ptr = address(x)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            out = x * 2
        release(x, side.record_event())
        del x
        y.fill_(7)
        self.assertEqual(address(y), ptr)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(out, torch.full_like(out, 2))
        self.assertEqual(y, torch.full_like(y, 7))

    def test_release_materialized(self, device):
        x = torch.ones(1 << 20, device=device)
        ptr = address(x)
        y = x._lazy_clone()
        x.add_(1)
        allocated = torch.cuda.memory_allocated(device)
        release(x)
        self.assertFalse(is_cow(x))
        y.add_(1)
        self.assertEqual(address(y), ptr)
        self.assertEqual(torch.cuda.memory_allocated(device), allocated)
        self.assertEqual(x, torch.full_like(x, 2))
        self.assertEqual(y, torch.full_like(y, 2))
        z = x._lazy_clone()
        self.assertTrue(is_cow(z))

    def test_unreleased_reference_prevents_steal(self, device):
        x = torch.ones(1 << 20, device=device)
        ptr = address(x)
        y = x._lazy_clone()
        z = x._lazy_clone()
        release(x)
        del x, y
        z.add_(1)
        self.assertNotEqual(address(z), ptr)

    def test_write_after_release(self, device):
        # Writing to y after releasing it reads the shared data once more,
        # which must complete before x steals it.
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        ptr = address(x)
        y = x._lazy_clone()
        release(y)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            y.add_(1)
        self.assertFalse(is_cow(y))
        x.fill_(5)
        self.assertEqual(address(x), ptr)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(x, torch.full_like(x, 5))
        self.assertEqual(y, torch.full_like(y, 2))

    def test_release_after_materialize(self, device):
        # The copy materializing x reads the shared data on the stream x is
        # written on, which releasing x cannot cover, so y's steal must still
        # wait for it.
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        ptr = address(x)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            x.add_(1)
        release(x)
        y.fill_(7)
        self.assertEqual(address(y), ptr)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(x, torch.full_like(x, 2))
        self.assertEqual(y, torch.full_like(y, 7))

    def test_free_after_release(self, device):
        # When the shared data is freed, its reuse must wait for the release
        # events.
        side = torch.cuda.Stream(device=device)
        x = torch.ones(1 << 20, device=device)
        y = x._lazy_clone()
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            torch.cuda._sleep(SLEEP_CYCLES)
            out = x * 2
        release(x, side.record_event())
        del x, y
        scratch = torch.empty(1 << 20, device=device)
        scratch.fill_(99)
        torch.cuda.current_stream(device).wait_stream(side)
        self.assertEqual(out, torch.full_like(out, 2))

    def test_release_event_types(self, device):
        for event in [torch.Event(device=device), torch.cuda.Event()]:
            x = torch.ones(16, device=device)
            ptr = address(x)
            y = x._lazy_clone()
            event.record()
            release(x, event)
            del x, event
            y.add_(1)
            self.assertEqual(address(y), ptr)

    def test_release_errors(self, device):
        x = torch.ones(16, device=device)
        y = x._lazy_clone()
        with self.assertRaisesRegex(RuntimeError, "expected a recorded event"):
            release(x, torch.cuda.Event())
        with self.assertRaisesRegex(RuntimeError, "expected a torch.Event"):
            release(x, 1)
        if TEST_CUDA_GRAPH:
            g = torch.cuda.CUDAGraph()
            with self.assertRaisesRegex(RuntimeError, "during CUDA graph capture"):
                with torch.cuda.graph(g):
                    release(y)
        # Releasing other tensors does nothing.
        release(torch.ones(16, device=device))
        cpu = torch.ones(16)
        cpu_clone = cpu._lazy_clone()
        release(cpu)
        self.assertTrue(is_cow(cpu_clone))

    @unittest.skipUnless(TEST_CUDA_GRAPH, "requires CUDA graphs")
    def test_lazy_clone_during_capture(self, device):
        # https://github.com/pytorch/pytorch/issues/200155
        static_in = torch.zeros(1 << 20, device=device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = static_in._lazy_clone()
            y.data_ptr()
            out = y * 2
        self.assertFalse(is_cow(y))
        for value in [3, 7]:
            static_in.fill_(value)
            g.replay()
            self.assertEqual(out, static_in * 2)

    @unittest.skipUnless(TEST_CUDA_GRAPH, "requires CUDA graphs")
    def test_materialize_during_capture_errors(self, device):
        x = torch.ones(16, device=device)
        y = x._lazy_clone()
        g = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "during CUDA graph capture"):
            with torch.cuda.graph(g):
                y.add_(1)
        self.assertTrue(is_cow(y))
        self.assertEqual(y, torch.ones_like(y))

    @unittest.skipUnless(TEST_CUDA_GRAPH, "requires CUDA graphs")
    def test_lazy_clone_of_graph_output_copies(self, device):
        # Replays write graph outputs, so they cannot be shared lazily, even on
        # the stream they were allocated on.
        stream = torch.cuda.Stream(device=device)
        static_in = torch.ones(16, device=device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            out = static_in * 2
        with torch.cuda.stream(stream):
            g.replay()
            snapshot = out._lazy_clone()
            self.assertFalse(is_cow(snapshot))
            static_in.fill_(3)
            g.replay()
        torch.cuda.synchronize(device)
        self.assertEqual(snapshot, torch.full_like(snapshot, 2))
        self.assertEqual(out, torch.full_like(out, 6))


instantiate_device_type_tests(TestCOWCUDA, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()

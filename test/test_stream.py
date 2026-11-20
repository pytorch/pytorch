# Owner(s): ["module: tests"]

import sys
import time
import unittest

import torch
from torch.testing._internal.common_device_type import (
    expectedFailureMPS,
    expectedFailureXPU,
    instantiate_device_type_tests,
    skipXPUIf,
)
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    HardwareClassification,
    NoTest,
    parametrize,
    run_tests,
    skipCUDANonDefaultStreamIf,
    subtest,
    TEST_ACCELERATOR,
    TEST_CUDA,
    TestCase,
)
from torch.testing._internal.common_xpu import Xe2_Or_Later


if not TEST_ACCELERATOR:
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest
    # Skip because failing when run on cuda build with no GPU, see #150059 for example
    sys.exit()

TEST_CUDAMALLOCASYNC = TEST_CUDA and (
    torch.cuda.get_allocator_backend() == "cudaMallocAsync"
)


class TestStream(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        m = torch.get_device_module(self.device_type)
        self._stream_reset = m.current_stream()

    def tearDown(self):
        m = torch.get_device_module(self.device_type)
        m.synchronize()
        m.empty_cache()
        m.set_stream(self._stream_reset)
        super().tearDown()

    def _get_stream_by_name(self, stream):
        m = torch.get_device_module(self.device_type)
        if stream == "default":
            if self.device_type == "xpu":
                self.skipTest("xpu does not have default_stream")
            return m.default_stream()
        elif stream == "user":
            return m.Stream()
        else:
            raise NotImplementedError(f"stream '{stream}' not recognized")

    # If use_generic_stream is set, create stream with `torch.Stream`.
    # Otherwise, use the module-specific `torch.<device>.Stream`.
    def _get_user_stream(self, use_generic_stream):
        if use_generic_stream:
            return torch.Stream(device=self.device_type)
        else:
            return torch.get_device_module(self.device_type).Stream()

    @parametrize(
        "use_generic_stream",
        # `torch.xpu.Stream` does not support comparison to `torch.Stream`
        [subtest(True, decorators=[expectedFailureXPU]), False],
    )
    def test_stream(self, device, use_generic_stream):
        m = torch.get_device_module(device)
        s0 = self._get_user_stream(use_generic_stream)
        s1 = self._get_user_stream(use_generic_stream)
        self.assertNotEqual(s0, s1)

        if self.device_type == "cuda":
            self.assertEqual(m.default_stream().cuda_stream, 0)
            self.assertNotEqual(s0.cuda_stream, 0)

        with m.stream(s0):
            self.assertEqual(m.current_stream(), s0)

        prev_stream = m.current_stream()
        with m.stream(None):
            self.assertEqual(m.current_stream(), prev_stream)

    @parametrize(
        "use_generic_stream",
        # `torch.xpu.Stream` does not support comparison to `torch.Stream`
        [subtest(True, decorators=[expectedFailureXPU]), False],
    )
    def test_set_stream(self, device, use_generic_stream):
        m = torch.get_device_module(device)
        s0 = self._get_user_stream(use_generic_stream)
        m.set_stream(s0)
        s1 = m.current_stream()
        self.assertEqual(s0, s1)
        s2 = self._get_user_stream(use_generic_stream)
        self.assertFalse(s0 == s2)
        m.set_stream(s2)
        with m.stream(s0):
            self.assertEqual(s0, m.current_stream())
        self.assertEqual(s2, m.current_stream())

    def test_stream_repr(self, device):
        m = torch.get_device_module(device)
        s = m.current_stream()
        self.assertTrue(f"torch.{self.device_type}.Stream" in str(s))

    # torch.mps.Event.__repr__ does not produce the correct output
    @expectedFailureMPS
    def test_stream_event_repr(self, device):
        m = torch.get_device_module(device)
        e = m.Event()
        self.assertTrue(f"torch.{self.device_type}.Event" in str(e))
        m.current_stream().record_event(e)
        self.assertTrue(f"torch.{self.device_type}.Event" in str(e))

    @skipCUDANonDefaultStreamIf(True)
    @parametrize("use_generic_stream", [True, False])
    def test_stream_query(self, device, use_generic_stream):
        m = torch.get_device_module(device)
        current_stream = m.current_stream()
        user_stream = self._get_user_stream(use_generic_stream)
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        current_stream.synchronize()
        self.assertTrue(current_stream.query())
        del tensor1

    # Check that `query` is False when the device is busy. A device-side sleep
    # for 200 ms should give the host plenty of time to query.
    @skipCUDANonDefaultStreamIf(True)
    @skipXPUIf(not Xe2_Or_Later, "torch.xpu._sleep is not supported")
    def test_stream_query_device_busy(self, device):
        m = torch.get_device_module(device)
        m._sleep(int(200 * get_cycles_per_ms(device)))
        self.assertFalse(m.current_stream().query())
        m.current_stream().synchronize()
        self.assertTrue(m.current_stream().query())

    # Makes the current stream wait on an event that is recorded after a short
    # device-side sleep on another stream
    def _wait_on_busy_stream(device):
        if torch.device(device).type == "xpu" and not Xe2_Or_Later:
            raise unittest.SkipTest("torch.xpu._sleep is not supported")
        m = torch.get_device_module(device)
        s = m.Stream()
        with m.stream(s):
            m._sleep(int(50 * get_cycles_per_ms(device)))
            e = s.record_event()
        m.current_stream().wait_event(e)
        return s, e

    # Check that `query` is True when the device finishes work, even if
    # `synchronize` has not been called yet.
    @skipCUDANonDefaultStreamIf(True)
    @parametrize(
        "case",
        [
            # Test querying for several different types of small workloads.
            # (init_fn, case_fn)
            (lambda device: [torch.randn(1, device=device)], lambda x: x + 1),
            (
                lambda device: [
                    torch.randn(8, 8, device=device),
                    torch.randn(8, 8, device=device),
                ],
                torch.mm,
            ),
            # Host-to-device copy
            (
                lambda device: [torch.randn(1024, device="cpu"), device],
                lambda x, device: x.to(device, non_blocking=True),
            ),
            # Wait on an event that has not yet been signaled
            (lambda device: [device], _wait_on_busy_stream),
        ],
    )
    def test_stream_query_no_sync(self, device, case):
        m = torch.get_device_module(device)
        init_fn, case_fn = case
        init_state = init_fn(device)

        m.current_stream().synchronize()
        final_state = case_fn(*init_state)

        done = m.current_stream().query()
        # Sleeping for 200 ms on host should give the device plenty of time to
        # finish.
        time.sleep(0.2)
        done = done or m.current_stream().query()
        self.assertTrue(done)
        del final_state

    def test_stream_query_after_event_sync(self, device):
        m = torch.get_device_module(device)
        e = m.Event()
        x = torch.randn(1, device=device)
        y = x + 1
        e.record()
        e.synchronize()
        self.assertTrue(m.current_stream().query())
        del y

    # Check that the stream is reset correctly if an exception is raised within
    # another stream's context.
    @parametrize(
        "use_generic_stream",
        # `torch.xpu.Stream` does not support comparison to `torch.Stream`
        [subtest(True, decorators=[expectedFailureXPU]), False],
    )
    def test_exception_reset(self, device, use_generic_stream):
        m = torch.get_device_module(device)
        s1 = self._get_user_stream(use_generic_stream)

        # Note: Use a custom exception type so that any unexpected errors inside
        # the `try` block pass through to fail the test
        class MyException(Exception):
            pass

        prev_stream = m.current_stream()

        try:
            with m.stream(s1):
                self.assertEqual(m.current_stream(), s1)
                raise MyException

        except MyException:
            self.assertEqual(m.current_stream(), prev_stream)

    @parametrize("stream_init", ["default", "user"])
    @parametrize(
        "use_generic_stream",
        # `torch.xpu.Stream` does not support comparison to `torch.Stream`
        [subtest(True, decorators=[expectedFailureXPU]), False],
    )
    def test_stream_context_manager(self, device, stream_init, use_generic_stream):
        m = torch.get_device_module(device)
        m.set_stream(self._get_stream_by_name(stream_init))
        prev_stream = m.current_stream()
        with self._get_user_stream(use_generic_stream) as stream:
            self.assertEqual(stream, m.current_stream())
        self.assertEqual(prev_stream, m.current_stream())

    # Check that the stream is reset correctly with nested contexts
    @parametrize(
        "use_generic_stream",
        # `torch.xpu.Stream` does not support comparison to `torch.Stream`
        [subtest(True, decorators=[expectedFailureXPU]), False],
    )
    def test_nested_reset(self, device, use_generic_stream):
        m = torch.get_device_module(device)
        s0 = m.current_stream()
        s1 = self._get_user_stream(use_generic_stream)
        s2 = self._get_user_stream(use_generic_stream)

        with m.stream(s1):
            self.assertEqual(m.current_stream(), s1)
            with m.stream(s2):
                self.assertEqual(m.current_stream(), s2)

            self.assertEqual(m.current_stream(), s1)

        self.assertEqual(m.current_stream(), s0)

    # torch.cpu.default_stream is not supported
    @expectedFailureXPU
    @skipCUDANonDefaultStreamIf(True)
    def test_default_stream(self, device):
        m = torch.get_device_module(device)
        self.assertEqual(m.current_stream(), m.default_stream())
        self.assertNotEqual(m.Stream(), m.default_stream())

    # Check that two separate workloads run on separate streams gives the same
    # result as running them on one stream.
    @parametrize("use_generic_stream", [True, False])
    def test_separate_workloads(self, device, use_generic_stream):
        m = torch.get_device_module(device)
        s1 = self._get_user_stream(use_generic_stream)
        s2 = self._get_user_stream(use_generic_stream)

        def concurrent_sine_loop(a1, a2, s1=None, s2=None, n=100):
            r1 = a1
            r2 = a2

            for _ in range(n):
                with m.stream(s1):
                    r1 = torch.sin(r1)
                with m.stream(s2):
                    r2 = torch.sin(r2)

            return r1, r2

        a1 = torch.randn(100, device=device)
        a2 = torch.randn(100, device=device)
        m.synchronize()

        r1, r2 = concurrent_sine_loop(a1, a2, s1, s2)
        s1.synchronize()
        s2.synchronize()

        r1_check, r2_check = concurrent_sine_loop(a1, a2, s1=None, s2=None)
        m.synchronize()

        self.assertEqual(r1, r1_check)
        self.assertEqual(r2, r2_check)

    @skipXPUIf(not Xe2_Or_Later, "torch.xpu._sleep is not supported")
    def test_record_stream(self, device):
        m = torch.get_device_module(device)
        cycles_per_ms = get_cycles_per_ms(device)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float).pin_memory()
        result = torch.empty(t.size(), dtype=torch.float, device=device)
        stream = m.Stream()
        ptr = [None]

        # Performs the CPU->GPU copy in a background stream
        def perform_copy():
            with m.stream(stream):
                tmp = t.to(device=device, non_blocking=True)
                ptr[0] = tmp.data_ptr()
            m.current_stream().wait_stream(stream)
            tmp.record_stream(m.current_stream())
            m._sleep(int(50 * cycles_per_ms))  # delay the copy
            result.copy_(tmp)

        perform_copy()
        with m.stream(stream):
            tmp2 = torch.empty(t.size(), dtype=torch.float, device=device)
            tmp2.zero_()
            self.assertNotEqual(
                tmp2.data_ptr(), ptr[0], msg="allocation reused to soon"
            )

        self.assertEqual(result.tolist(), [1, 2, 3, 4])

        if (
            self.device_type == "cuda" and not TEST_CUDAMALLOCASYNC
        ) or self.device_type == "mps":
            # In the native CUDA allocator and MPS allocator, we expect "tmp"'s
            # side-stream-tagged block will be reused in that side stream after
            # result.copy_(tmp) in the main stream finishes.
            m.current_stream().synchronize()
            with m.stream(stream):
                tmp3 = torch.empty(t.size(), dtype=torch.float, device=device)
                self.assertEqual(tmp3.data_ptr(), ptr[0], msg="allocation not reused")


instantiate_device_type_tests(
    TestStream,
    globals(),
    except_for=("cpu",),
    allow_mps=True,
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()

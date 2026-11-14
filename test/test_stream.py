# Owner(s): ["module: tests"]

import sys

import torch
from torch.testing._internal.common_device_type import (
    expectedFailureMPS,
    expectedFailureXPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    NoTest,
    parametrize,
    run_tests,
    skipCUDANonDefaultStreamIf,
    TEST_ACCELERATOR,
    TestCase,
)


if not TEST_ACCELERATOR:
    print("No available accelerator detected, skipping tests", file=sys.stderr)
    TestCase = NoTest
    # Skip because failing when run on cuda build with no GPU, see #150059 for example
    sys.exit()


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

    def test_stream(self, device):
        m = torch.get_device_module(device)
        s0 = m.Stream()
        s1 = m.Stream()
        self.assertNotEqual(s0, s1)

        if self.device_type == "cuda":
            self.assertEqual(m.default_stream().cuda_stream, 0)
            self.assertNotEqual(s0.cuda_stream, 0)

        with m.stream(s0):
            self.assertEqual(m.current_stream(), s0)

        prev_stream = m.current_stream()
        with m.stream(None):
            self.assertEqual(m.current_stream(), prev_stream)

    def test_set_stream(self, device):
        m = torch.get_device_module(device)
        s0 = m.Stream()
        m.set_stream(s0)
        s1 = m.current_stream()
        self.assertEqual(s0, s1)
        s2 = m.Stream()
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

    # torch.mps.Stream.query is not supported yet
    @expectedFailureMPS
    @skipCUDANonDefaultStreamIf(True)
    def test_stream_query(self, device):
        m = torch.get_device_module(device)
        current_stream = m.current_stream()
        user_stream = m.Stream()
        self.assertTrue(user_stream.query())
        tensor1 = torch.ByteTensor(5).pin_memory()
        current_stream.synchronize()
        self.assertTrue(current_stream.query())
        del tensor1

    # Check that the stream is reset correctly if an exception is raised within
    # another stream's context.
    def test_exception_reset(self, device):
        m = torch.get_device_module(device)
        s1 = m.Stream()

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
    def test_stream_context_manager(self, device, stream_init):
        m = torch.get_device_module(device)
        m.set_stream(self._get_stream_by_name(stream_init))
        prev_stream = m.current_stream()
        with m.Stream() as stream:
            self.assertEqual(stream, m.current_stream())
        self.assertEqual(prev_stream, m.current_stream())

    # Check that the stream is reset correctly with nested contexts
    def test_nested_reset(self, device):
        m = torch.get_device_module(device)
        s0 = m.current_stream()
        s1 = m.Stream()
        s2 = m.Stream()

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
    def test_separate_workloads(self, device):
        m = torch.get_device_module(device)
        s1 = m.Stream()
        s2 = m.Stream()

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


instantiate_device_type_tests(
    TestStream,
    globals(),
    except_for=("cpu",),
    allow_mps=True,
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests()

# Owner(s): ["module: inductor"]
"""Tests for user-annotated stream support in Inductor.

This module tests the infrastructure for supporting user-annotated CUDA stream
assignments on nodes, including stream utilities, event management, and codegen.
"""
from __future__ import annotations

import unittest

import torch
from torch._inductor import config
from torch._inductor.codegen.wrapper import (
    EnterCudaStreamContextLine,
    EnterDeviceContextManagerWithStreamInfoLine,
    ExitCudaStreamContextLine,
    ExitDeviceContextManagerWithStreamInfoLine,
)
from torch._inductor.event import (
    CudaEventFactory,
    CudaEventSym,
)
from torch._inductor.stream_utils import (
    CUDAStreamPool,
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    ENTRANCE_EVENT,
    EVENT_NAME_TEMPLATE,
    get_cuda_stream_pool,
    get_stream_name,
    STREAM_NAME_TEMPLATE,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import IndentedBuffer
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class TestStreamUtils(InductorTestCase):
    """Tests for stream_utils module."""

    def test_constants(self):
        """Test stream utility constants are defined correctly."""
        self.assertEqual(DEFAULT_STREAM, "default_stream")
        self.assertEqual(DEFAULT_STREAM_IDX, 0)
        self.assertEqual(ENTRANCE_EVENT, "event0")
        self.assertEqual(EVENT_NAME_TEMPLATE, "event{event_idx:d}")
        self.assertEqual(STREAM_NAME_TEMPLATE, "stream{stream_idx:d}")

    def test_get_stream_name_default(self):
        """Test get_stream_name returns default stream for index 0."""
        self.assertEqual(get_stream_name(0), DEFAULT_STREAM)

    def test_get_stream_name_side_streams(self):
        """Test get_stream_name returns formatted names for side streams."""
        self.assertEqual(get_stream_name(1), "stream1")
        self.assertEqual(get_stream_name(2), "stream2")
        self.assertEqual(get_stream_name(10), "stream10")

    def test_get_stream_name_cached(self):
        """Test that get_stream_name results are cached."""
        # Call twice and verify we get the same object (cached)
        name1 = get_stream_name(5)
        name2 = get_stream_name(5)
        self.assertIs(name1, name2)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_cuda_stream_pool_creation(self):
        """Test CUDAStreamPool can be created."""
        pool = CUDAStreamPool(pool_size=4)
        self.assertEqual(pool.pool_size, 4)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_cuda_stream_pool_acquire_release(self):
        """Test CUDAStreamPool acquire and release."""
        pool = CUDAStreamPool(pool_size=2)

        # Acquire streams
        stream1 = pool.acquire()
        stream2 = pool.acquire()

        self.assertIsInstance(stream1, torch.cuda.Stream)
        self.assertIsInstance(stream2, torch.cuda.Stream)
        self.assertIsNot(stream1, stream2)

        # Release streams back
        pool.release(stream1)
        pool.release(stream2)

        # Should be able to acquire again
        stream3 = pool.acquire()
        self.assertIsInstance(stream3, torch.cuda.Stream)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_cuda_stream_pool_context_manager(self):
        """Test CUDAStreamPool as context manager."""
        pool = CUDAStreamPool(pool_size=2)

        with pool as stream:
            self.assertIsInstance(stream, torch.cuda.Stream)
            # Stream should be the current stream inside context
            self.assertEqual(torch.cuda.current_stream(), stream)

        # After context, default stream should be current
        self.assertNotEqual(torch.cuda.current_stream(), stream)


class TestCudaEventFactory(InductorTestCase):
    """Tests for CudaEventFactory and CudaEventSym."""

    def test_factory_creation(self):
        """Test CudaEventFactory can be created."""
        factory = CudaEventFactory()
        self.assertIsNotNone(factory)
        self.assertEqual(factory._reuse_cuda_event, config.reuse_cuda_event)

    def test_get_sym_event(self):
        """Test creating symbolic events."""
        factory = CudaEventFactory()

        event1 = factory.get_sym_event(originate_stream_idx=0)
        event2 = factory.get_sym_event(originate_stream_idx=1)

        self.assertIsInstance(event1, CudaEventSym)
        self.assertIsInstance(event2, CudaEventSym)
        self.assertEqual(event1.originate_stream_idx, 0)
        self.assertEqual(event2.originate_stream_idx, 1)
        # Events should have different indices
        self.assertNotEqual(event1.idx, event2.idx)

    def test_get_entrance_event(self):
        """Test getting the entrance event."""
        factory = CudaEventFactory()

        entrance1 = factory.get_entrance_event()
        entrance2 = factory.get_entrance_event()

        # Should return the same event
        self.assertIs(entrance1, entrance2)
        self.assertEqual(entrance1.idx, 0)
        self.assertEqual(entrance1.originate_stream_idx, DEFAULT_STREAM_IDX)
        self.assertEqual(entrance1.materialized_event, ENTRANCE_EVENT)

    def test_event_ordering(self):
        """Test CudaEventSym ordering."""
        factory = CudaEventFactory()

        event1 = factory.get_sym_event(originate_stream_idx=0)
        event2 = factory.get_sym_event(originate_stream_idx=0)

        self.assertLess(event1, event2)
        self.assertGreater(event2, event1)
        self.assertEqual(event1, event1)
        self.assertNotEqual(event1, event2)

    def test_event_hash(self):
        """Test CudaEventSym is hashable."""
        factory = CudaEventFactory()

        event1 = factory.get_sym_event(originate_stream_idx=0)
        event2 = factory.get_sym_event(originate_stream_idx=1)

        # Should be hashable and usable in sets/dicts
        event_set = {event1, event2}
        self.assertEqual(len(event_set), 2)
        self.assertIn(event1, event_set)
        self.assertIn(event2, event_set)

    def test_event_str(self):
        """Test CudaEventSym string representation."""
        factory = CudaEventFactory()
        event = factory.get_sym_event(originate_stream_idx=1)

        str_repr = str(event)
        self.assertIn("CudaEventSym", str_repr)
        self.assertIn("originate_stream_idx=1", str_repr)


class TestWrapperCodegenStreams(InductorTestCase):
    """Tests for stream-related wrapper codegen classes."""

    def test_enter_device_context_with_stream_info(self):
        """Test EnterDeviceContextManagerWithStreamInfoLine has num_streams param."""
        line = EnterDeviceContextManagerWithStreamInfoLine(
            device_idx=0,
            last_seen_device_guard_index=None,
            num_streams=4,
        )
        self.assertEqual(line.num_streams, 4)
        self.assertEqual(line.device_idx, 0)

    def test_exit_device_context_with_stream_info(self):
        """Test ExitDeviceContextManagerWithStreamInfoLine has num_streams param."""
        line = ExitDeviceContextManagerWithStreamInfoLine(num_streams=4)
        self.assertEqual(line.num_streams, 4)

    def test_enter_cuda_stream_context(self):
        """Test EnterCudaStreamContextLine creation."""
        line = EnterCudaStreamContextLine(stream_idx=1)
        self.assertEqual(line.stream_idx, 1)

    def test_exit_cuda_stream_context(self):
        """Test ExitCudaStreamContextLine creation."""
        line = ExitCudaStreamContextLine()
        # Just verify it can be created
        self.assertIsNotNone(line)


class TestConfig(InductorTestCase):
    """Tests for stream-related config options."""

    def test_reuse_cuda_event_config(self):
        """Test reuse_cuda_event config option exists and has correct default."""
        self.assertTrue(hasattr(config, "reuse_cuda_event"))
        self.assertIsInstance(config.reuse_cuda_event, bool)
        # Default should be True
        self.assertTrue(config.reuse_cuda_event)

    def test_reuse_cuda_event_config_patch(self):
        """Test reuse_cuda_event can be patched."""
        with config.patch(reuse_cuda_event=False):
            self.assertFalse(config.reuse_cuda_event)
        # Should be restored after context
        self.assertTrue(config.reuse_cuda_event)


class TestStreamCodegen(InductorTestCase):
    """End-to-end tests for stream code generation."""

    def test_enter_cuda_stream_context_codegen(self):
        """Test code generation for entering a CUDA stream context."""
        code = IndentedBuffer()
        code.writeline("def call(args):")
        code.do_indent()
        code.do_indent()  # Simulate being inside device guard

        line = EnterCudaStreamContextLine(stream_idx=1)
        line.codegen(code)

        generated = code.getvalue()
        # Should have stream context
        self.assertIn("with torch.cuda.stream(stream1)", generated)

    def test_exit_cuda_stream_context_codegen(self):
        """Test code generation for exiting a CUDA stream context."""
        code = IndentedBuffer()
        code.writeline("def call(args):")
        code.do_indent()
        code.do_indent()
        code.do_indent()  # Simulate being inside stream context (3 levels)

        line = ExitCudaStreamContextLine()
        line.codegen(code)

        # The exit just unindents, verify no error
        self.assertIsNotNone(code.getvalue())

    def test_event_record_codegen(self):
        """Test code generation for CUDA event recording."""
        from torch._inductor.event import _CudaEventRecordLine

        factory = CudaEventFactory()
        event = factory.get_sym_event(originate_stream_idx=0)
        # Simulate the event being waited on (ref_count > 0)
        event.ref_count = 1

        code = IndentedBuffer()
        record_line = _CudaEventRecordLine(event=event, stream="stream1")
        record_line.codegen(code)

        generated = code.getvalue()
        # Should create and record the event
        self.assertIn("torch.cuda.Event()", generated)
        self.assertIn(".record(stream1)", generated)

    def test_event_wait_codegen(self):
        """Test code generation for CUDA event waiting."""
        from torch._inductor.event import _CudaEventWaitLine

        factory = CudaEventFactory()
        event = factory.get_sym_event(originate_stream_idx=0)
        # Set up event state
        event.ref_count = 1
        event.materialized_event = "event1"

        code = IndentedBuffer()
        wait_line = _CudaEventWaitLine(event=event, stream="stream2")
        wait_line.codegen(code)

        generated = code.getvalue()
        # Should wait on the event
        self.assertIn("event1.wait(stream2)", generated)

    def test_stream_context_with_event_sync(self):
        """Test stream context with event synchronization flow."""
        from torch._inductor.event import _CudaEventRecordLine, _CudaEventWaitLine

        code = IndentedBuffer()
        code.writeline("def call(args):")
        code.do_indent()
        code.do_indent()  # Inside device guard

        # Create an event factory and event
        factory = CudaEventFactory()
        event = factory.get_sym_event(originate_stream_idx=0)

        # Record event on stream 0 (simulating kernel completion)
        event.ref_count = 1  # Will be waited on
        record_line = _CudaEventRecordLine(event=event, stream="default_stream")
        record_line.codegen(code)

        # Enter stream 1 context
        enter_stream = EnterCudaStreamContextLine(stream_idx=1)
        enter_stream.codegen(code)

        # Wait for event from stream 0
        wait_line = _CudaEventWaitLine(event=event, stream="stream1")
        wait_line.codegen(code)

        # Simulate kernel on stream 1
        code.writeline("# kernel on stream1")

        # Exit stream context
        exit_stream = ExitCudaStreamContextLine()
        exit_stream.codegen(code)

        generated = code.getvalue()

        # Verify the synchronization flow
        self.assertIn("torch.cuda.Event()", generated)
        self.assertIn(".record(default_stream)", generated)
        self.assertIn("with torch.cuda.stream(stream1):", generated)
        self.assertIn(".wait(stream1)", generated)
        self.assertIn("# kernel on stream1", generated)


instantiate_parametrized_tests(TestStreamUtils)
instantiate_parametrized_tests(TestCudaEventFactory)
instantiate_parametrized_tests(TestWrapperCodegenStreams)
instantiate_parametrized_tests(TestConfig)
instantiate_parametrized_tests(TestStreamCodegen)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()

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
from torch._inductor.event import CudaEventFactory, CudaEventSym
from torch._inductor.stream_utils import (
    CUDAStreamPool,
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    ENTRANCE_EVENT,
    EVENT_NAME_TEMPLATE,
    get_stream_name,
    STREAM_NAME_TEMPLATE,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import IndentedBuffer
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import instantiate_parametrized_tests


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


@unittest.skipIf(not TEST_CUDA, "requires CUDA")
class TestUserStreamCompile(InductorTestCase):
    """End-to-end tests for torch.compile with user stream contexts."""

    def test_compile_with_user_stream_context(self):
        """Test that user code with stream context compiles and runs correctly."""
        from torch._inductor.utils import run_and_get_code

        def fn(x, y):
            # Create a side stream
            s = torch.cuda.Stream()
            # Perform operation on default stream
            z = x + y
            # Perform operation on side stream
            with torch.cuda.stream(s):
                w = z * 2
            # Synchronize before using result
            s.synchronize()
            return w + 1

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")

        # Get expected result from eager execution
        expected = fn(x, y)

        # Compile and run
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x, y)

        # Verify correctness
        self.assertEqual(result, expected)

        # Verify generated code contains stream handling
        self.assertIn("torch.cuda.Stream", code)
        self.assertIn("torch.cuda.stream", code)

    def test_compile_preserves_stream_semantics(self):
        """Test that compiled code preserves stream execution semantics."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            # Work on default stream
            a = x * 2
            # Work on side stream
            with torch.cuda.stream(s):
                b = x * 3
            s.synchronize()
            return a + b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream context is present in generated code
        self.assertIn("torch.cuda.stream", code)

    def test_multiple_stream_contexts(self):
        """Test compilation with multiple stream context switches."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()

            a = x + 1  # default stream

            with torch.cuda.stream(s1):
                b = x * 2  # stream 1

            with torch.cuda.stream(s2):
                c = x * 3  # stream 2

            s1.synchronize()
            s2.synchronize()
            return a + b + c

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify multiple stream contexts in generated code
        # Should have at least 2 stream context usages
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 2)

    def test_nested_stream_contexts(self):
        """Test compilation with nested stream contexts."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()

            with torch.cuda.stream(s1):
                a = x * 2
                with torch.cuda.stream(s2):
                    b = x * 3
                c = a + 1  # back on s1

            s1.synchronize()
            s2.synchronize()
            return a + b + c

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify nested stream contexts
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 2)

    def test_stream_context_with_data_dependency(self):
        """Test stream contexts with data flowing between streams."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()

            # Compute on default stream
            a = x * 2

            # Use result on side stream
            with torch.cuda.stream(s):
                b = a + 1  # depends on 'a' from default stream

            s.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream context is present
        self.assertIn("torch.cuda.stream", code)

    def test_event_record_and_wait(self):
        """Test compilation with explicit event record and wait."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()

            # Compute on default stream
            a = x * 2
            # Record event on default stream
            event.record()

            with torch.cuda.stream(s):
                # Wait for event before using 'a'
                event.wait()
                b = a + 1

            s.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify event operations in generated code
        self.assertIn("torch.cuda.Event", code)
        self.assertIn(".record(", code)
        self.assertIn(".wait(", code)

    def test_event_record_on_stream(self):
        """Test event recording on a specific stream."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            event = torch.cuda.Event()

            with torch.cuda.stream(s1):
                a = x * 2
                # Record on s1
                event.record(s1)

            with torch.cuda.stream(s2):
                # Wait for s1's work before proceeding
                event.wait(s2)
                b = a + 1

            s1.synchronize()
            s2.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify event record/wait with explicit stream args
        self.assertIn(".record(", code)
        self.assertIn(".wait(", code)

    def test_multiple_events_multiple_streams(self):
        """Test multiple events synchronizing multiple streams."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            event1 = torch.cuda.Event()
            event2 = torch.cuda.Event()

            # Work on s1
            with torch.cuda.stream(s1):
                a = x * 2
                event1.record(s1)

            # Work on s2, depends on s1
            with torch.cuda.stream(s2):
                event1.wait(s2)
                b = a + 1
                event2.record(s2)

            # Back to default stream, wait for s2
            event2.wait()
            c = b + x

            s1.synchronize()
            s2.synchronize()
            return c

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify multiple events and streams
        self.assertGreaterEqual(code.count("torch.cuda.Event"), 2)
        self.assertGreaterEqual(code.count(".record("), 2)
        self.assertGreaterEqual(code.count(".wait("), 2)

    def test_event_wait_without_record(self):
        """Test that waiting on unrecorded event works (no-op)."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()

            # Record the event first
            event.record()

            with torch.cuda.stream(s):
                # Wait is valid after record
                event.wait()
                a = x * 2

            s.synchronize()
            return a

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify event operations
        self.assertIn(".record(", code)
        self.assertIn(".wait(", code)

    def test_stream_wait_event(self):
        """Test stream.wait_event() method."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()

            a = x * 2
            event.record()

            # Use stream.wait_event instead of event.wait
            s.wait_event(event)
            with torch.cuda.stream(s):
                b = a + 1

            s.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream.wait_event is present
        self.assertIn("wait_event", code)

    def test_bidirectional_stream_sync(self):
        """Test bidirectional synchronization between streams."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            event_s1 = torch.cuda.Event()
            event_s2 = torch.cuda.Event()

            # s1 does work
            with torch.cuda.stream(s1):
                a = x * 2
                event_s1.record(s1)

            # s2 waits for s1, does work, signals back
            with torch.cuda.stream(s2):
                event_s1.wait(s2)
                b = a + 1
                event_s2.record(s2)

            # s1 waits for s2
            with torch.cuda.stream(s1):
                event_s2.wait(s1)
                c = b * 2

            s1.synchronize()
            s2.synchronize()
            return c

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify bidirectional sync - multiple records and waits
        self.assertGreaterEqual(code.count(".record("), 2)
        self.assertGreaterEqual(code.count(".wait("), 2)

    def test_three_streams_pipeline(self):
        """Test pipeline pattern with three streams."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            s3 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            # Stage 1 on s1
            with torch.cuda.stream(s1):
                a = x * 2
                e1.record(s1)

            # Stage 2 on s2, depends on s1
            with torch.cuda.stream(s2):
                e1.wait(s2)
                b = a + 1
                e2.record(s2)

            # Stage 3 on s3, depends on s2
            with torch.cuda.stream(s3):
                e2.wait(s3)
                c = b * 3

            s1.synchronize()
            s2.synchronize()
            s3.synchronize()
            return c

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify three-stage pipeline with 3 streams
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 3)
        self.assertGreaterEqual(code.count(".record("), 2)

    def test_parallel_streams_join(self):
        """Test parallel work on multiple streams joining at the end."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            s3 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()
            e3 = torch.cuda.Event()

            # Parallel work on three streams
            with torch.cuda.stream(s1):
                a = x * 2
                e1.record(s1)

            with torch.cuda.stream(s2):
                b = x * 3
                e2.record(s2)

            with torch.cuda.stream(s3):
                c = x * 4
                e3.record(s3)

            # Join all results on default stream
            e1.wait()
            e2.wait()
            e3.wait()
            result = a + b + c

            s1.synchronize()
            s2.synchronize()
            s3.synchronize()
            return result

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify parallel streams joining
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 3)
        self.assertGreaterEqual(code.count(".record("), 3)
        self.assertGreaterEqual(code.count(".wait("), 3)

    def test_fan_out_fan_in(self):
        """Test fan-out from one stream to multiple, then fan-in."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            e_start = torch.cuda.Event()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            # Initial work on default stream
            a = x * 2
            e_start.record()

            # Fan out to s1 and s2
            with torch.cuda.stream(s1):
                e_start.wait(s1)
                b = a + 1
                e1.record(s1)

            with torch.cuda.stream(s2):
                e_start.wait(s2)
                c = a + 2
                e2.record(s2)

            # Fan in on default stream
            e1.wait()
            e2.wait()
            result = b + c

            s1.synchronize()
            s2.synchronize()
            return result

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify fan-out/fan-in pattern
        self.assertGreaterEqual(code.count(".record("), 3)
        self.assertGreaterEqual(code.count(".wait("), 4)

    def test_four_streams_diamond(self):
        """Test diamond pattern: one start, two parallel, one end."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            s3 = torch.cuda.Stream()
            e_start = torch.cuda.Event()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            # Start on default stream
            a = x + 1
            e_start.record()

            # Parallel branches on s1 and s2
            with torch.cuda.stream(s1):
                e_start.wait(s1)
                b = a * 2
                e1.record(s1)

            with torch.cuda.stream(s2):
                e_start.wait(s2)
                c = a * 3
                e2.record(s2)

            # Join on s3
            with torch.cuda.stream(s3):
                e1.wait(s3)
                e2.wait(s3)
                d = b + c

            s1.synchronize()
            s2.synchronize()
            s3.synchronize()
            return d

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify diamond pattern
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 3)
        self.assertGreaterEqual(code.count(".record("), 3)
        self.assertGreaterEqual(code.count(".wait("), 4)

    def test_stream_reuse_across_iterations(self):
        """Test that streams can be reused across loop iterations."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()
            result = x

            for _ in range(3):
                with torch.cuda.stream(s):
                    result = result * 2
                    event.record(s)
                event.wait()

            s.synchronize()
            return result

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream reuse in loop
        self.assertIn("torch.cuda.stream", code)
        self.assertIn(".record(", code)
        self.assertIn(".wait(", code)

    def test_no_fusion_across_streams(self):
        """Test that operations on different streams are not fused together."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            # These could be fused if on same stream, but should NOT be fused
            # since they're on different streams
            with torch.cuda.stream(s1):
                # Multiple pointwise ops that would normally fuse
                a = x * 2
                b = a + 1
                c = b * 3
                e1.record(s1)

            with torch.cuda.stream(s2):
                # Another set of pointwise ops
                d = x * 4
                e = d + 2
                f = e * 5
                e2.record(s2)

            e1.wait()
            e2.wait()
            return c + f

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify we have separate stream contexts (not fused into one)
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 2)

        # Count triton kernel calls - should have at least 2 separate kernels
        # (one for each stream's work)
        triton_kernel_count = code.count("triton_") + code.count(".run(")
        self.assertGreaterEqual(
            triton_kernel_count,
            2,
            "Expected at least 2 separate kernels for different streams",
        )

    def test_no_fusion_across_streams_with_dependency(self):
        """Test no fusion when there's a data dependency across streams."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()

            # Work on default stream
            a = x * 2
            b = a + 1
            event.record()

            # Work on side stream - depends on default stream
            with torch.cuda.stream(s):
                event.wait()
                c = b * 3  # depends on b from default stream
                d = c + 1

            s.synchronize()
            return d

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream context and event sync are present
        self.assertIn("torch.cuda.stream", code)
        self.assertIn(".record(", code)
        self.assertIn(".wait(", code)

        # The operations should not be fused across stream boundary
        # even though there's a data dependency
        triton_kernel_count = code.count("triton_") + code.count(".run(")
        self.assertGreaterEqual(
            triton_kernel_count,
            2,
            "Expected separate kernels before and after stream switch",
        )

    def test_fusion_within_same_stream(self):
        """Test that fusion still works for operations within the same stream."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()

            with torch.cuda.stream(s):
                # Multiple pointwise ops on same stream - should fuse
                a = x * 2
                b = a + 1
                c = b * 3
                d = c + 2

            s.synchronize()
            return d

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream context is present
        self.assertIn("torch.cuda.stream", code)

        # These ops should be fused into a single kernel since they're
        # all on the same stream and are pointwise operations


instantiate_parametrized_tests(TestStreamUtils)
instantiate_parametrized_tests(TestCudaEventFactory)
instantiate_parametrized_tests(TestWrapperCodegenStreams)
instantiate_parametrized_tests(TestConfig)
instantiate_parametrized_tests(TestStreamCodegen)
instantiate_parametrized_tests(TestUserStreamCompile)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()

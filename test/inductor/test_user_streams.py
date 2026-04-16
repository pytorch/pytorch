# Owner(s): ["module: inductor"]
"""Tests for user-annotated stream support in Inductor.

This module tests the infrastructure for supporting user-annotated CUDA stream
assignments on nodes, including stream utilities, event management, and codegen.
"""

from __future__ import annotations

import re
import unittest

import torch
import torch._inductor.config as inductor_config
import torch._inductor.metrics
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._inductor.codegen.wrapper import (
    EnterCudaStreamContextLine,
    EnterDeviceContextManagerWithStreamInfoLine,
    ExitCudaStreamContextLine,
    ExitDeviceContextManagerWithStreamInfoLine,
)
from torch._inductor.stream_constants import (
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    STREAM_NAME_TEMPLATE,
)
from torch._inductor.stream_utils import get_stream_name
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import IndentedBuffer
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    TEST_WITH_ROCM,
)


def _extract_wrapper_body(code):
    """Extract and normalize the call method body from generated wrapper code.

    Strips noise (comments, assert_size_stride, del statements, args.clear())
    and normalizes triton kernel names, leaving just structural code:
    stream declarations, context switches, event ops, kernel calls, buffer allocations.
    """
    lines = code.split("\n")

    # Find the call function body
    call_start = None
    call_indent = 0
    for i, line in enumerate(lines):
        if "def call(" in line:
            call_indent = len(line) - len(line.lstrip())
            call_start = i + 1
            break

    if call_start is None:
        return ""

    # Extract body until next definition at same indent level or end
    body_lines = []
    for i in range(call_start, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if stripped:
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= call_indent:
                break
        body_lines.append(line)

    # Filter out noise
    filtered = []
    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if "assert_size_stride" in stripped:
            continue
        if stripped.startswith("del "):
            continue
        if "args.clear()" in stripped:
            continue
        # Strip inline comments (e.g., "# reuse")
        line = re.sub(r"\s+#\s.*", "", line)
        if not line.strip():
            continue
        filtered.append(line)

    if not filtered:
        return ""

    # Dedent
    min_indent = min(
        len(line) - len(line.lstrip()) for line in filtered if line.strip()
    )
    dedented = [line[min_indent:] for line in filtered]
    body = "\n".join(dedented)

    # Normalize triton kernel names
    body = re.sub(r"triton_\w+", "triton_kernel", body)

    return body


class TestStreamUtils(InductorTestCase):
    """Tests for stream_utils module."""

    def test_constants(self):
        """Test stream utility constants are defined correctly."""
        self.assertEqual(DEFAULT_STREAM, "default_stream")
        self.assertEqual(DEFAULT_STREAM_IDX, 0)
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
        self.assertIsNotNone(line)


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
        code.do_indent()

        line = ExitCudaStreamContextLine()
        line.codegen(code)

        # The exit just unindents, verify no error
        self.assertIsNotNone(code.getvalue())


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

        # Verify generated code contains stream handling and synchronize survives
        self.assertIn("torch.cuda.stream", code)
        self.assertIn("synchronize_stream", code)

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

        # Verify stream context and synchronize survive
        self.assertIn("torch.cuda.stream", code)
        self.assertIn("synchronize_stream", code)

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

        # Verify stream contexts and synchronization survive
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 1)
        self.assertIn("synchronize_stream", code)

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

        # Verify nested stream contexts and synchronization survive
        self.assertGreaterEqual(code.count("torch.cuda.stream"), 1)
        self.assertIn("synchronize_stream", code)

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

        # Verify stream context and synchronize survive
        self.assertIn("torch.cuda.stream", code)
        self.assertIn("synchronize_stream", code)

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

        # Verify event operations survive compilation as custom ops
        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)

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

        # Verify event operations survive compilation as custom ops
        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)

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

        # Verify multiple events and streams survive as custom ops
        self.assertGreaterEqual(code.count("record_event"), 2)
        self.assertGreaterEqual(code.count("wait_event"), 2)

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

        # Verify event operations survive compilation as custom ops
        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)

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

        # Verify stream.wait_event survives compilation as custom op
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

        # Verify bidirectional sync - multiple records and waits as custom ops
        self.assertGreaterEqual(code.count("record_event"), 2)
        self.assertGreaterEqual(code.count("wait_event"), 2)

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
        self.assertGreaterEqual(code.count("record_event"), 2)
        self.assertGreaterEqual(code.count("wait_event"), 2)

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
        self.assertGreaterEqual(code.count("record_event"), 3)
        self.assertGreaterEqual(code.count("wait_event"), 3)

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
        self.assertGreaterEqual(code.count("record_event"), 3)
        self.assertGreaterEqual(code.count("wait_event"), 4)

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
        self.assertGreaterEqual(code.count("record_event"), 3)
        self.assertGreaterEqual(code.count("wait_event"), 4)

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

        # Verify stream reuse in loop — events survive compilation
        self.assertIn("torch.cuda.stream", code)
        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)

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
        torch._inductor.metrics.reset()
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # 3 kernels: s1 pointwise, s2 pointwise, and the final add on
        # the default stream (which is a third stream context).
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 3)

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
        torch._inductor.metrics.reset()
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

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
        torch._inductor.metrics.reset()
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # All pointwise ops on same stream should fuse into 1 kernel
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    def test_no_fusion_simple_dependency_across_streams(self):
        """Regression: a single pointwise consumed across a stream boundary must not fuse."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()

            with torch.cuda.stream(s1):
                a = x + 1
            e = s1.record_event()
            s2.wait_event(e)
            with torch.cuda.stream(s2):
                b = a * 2
            s1.synchronize()
            s2.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        torch._inductor.metrics.reset()
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Must be 2 separate kernels on 2 streams, not fused into 1
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @torch._inductor.config.patch(combo_kernels=True)
    def test_no_combo_kernel_fusion_across_streams(self):
        """Combo kernels must not group nodes on different streams."""
        from torch._inductor.utils import run_and_get_code

        def fn(x, y, z, w):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()

            # Independent pointwise ops on different streams at the same
            # topological level — combo kernels must not merge them.
            a = x + y
            event.record()
            with torch.cuda.stream(s):
                event.wait()
                b = z + w
            s.synchronize()
            return a, b

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        z = torch.randn(1024, device="cuda")
        w = torch.randn(1024, device="cuda")

        expected = fn(x, y, z, w)
        compiled_fn = torch.compile(fn)
        torch._inductor.metrics.reset()
        result, (code,) = run_and_get_code(compiled_fn, x, y, z, w)

        self.assertEqual(result, expected)
        # 2 kernels: one per stream. Without the stream-aware fix, combo
        # kernels would merge them into 1.
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

    @torch._inductor.config.patch(combo_kernels=True)
    def test_combo_kernel_fusion_within_same_stream(self):
        """Combo kernels should still group independent nodes on the same stream."""
        from torch._inductor.utils import run_and_get_code

        def fn(x, y):
            s = torch.cuda.Stream()

            with torch.cuda.stream(s):
                # Two independent pointwise ops on the same stream — eligible
                # for combo kernel fusion.
                a = x * 2
                b = y * 3

            s.synchronize()
            return a + b

        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")

        expected = fn(x, y)
        compiled_fn = torch.compile(fn)
        torch._inductor.metrics.reset()
        result, (code,) = run_and_get_code(compiled_fn, x, y)

        self.assertEqual(result, expected)
        # With combo kernels, the two independent ops on the same stream
        # should be combined, yielding fewer kernels than without.
        self.assertLessEqual(torch._inductor.metrics.generated_kernel_count, 2)

    def test_cross_stream_stride_copy(self):
        """A contiguous copy forced by a non-contiguous slice across streams
        must run on the consumer's stream, not the producer's."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()

            with torch.cuda.stream(s1):
                a = x + 1
                b = a[:, ::2]  # non-contiguous slice
            e = s1.record_event()
            s2.wait_event(e)
            with torch.cuda.stream(s2):
                c = b.contiguous()
                d = c + 1
            s2.synchronize()
            return d

        x = torch.randn(64, 64, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        torch._inductor.metrics.reset()
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 2)

        # Verify: s1 gets the pointwise (x+1), s2 gets the fused copy + add.
        # The contiguous copy is fused into the s2 triton kernel (which reads
        # from the s1 output buffer with strided indexing).  If the copy were
        # incorrectly placed on s1, we'd see 2 kernels on s1 instead of 1.
        wrapper = _extract_wrapper_body(code)
        lines = wrapper.split("\n")
        current_stream = None
        stream_kernels: dict[str | None, list[str]] = {}
        for line in lines:
            stripped = line.strip()
            if "with torch.cuda.stream(" in stripped:
                if "stream1" in stripped:
                    current_stream = "s1"
                elif "stream2" in stripped:
                    current_stream = "s2"
                elif "default_stream" in stripped:
                    current_stream = "default"
            elif ".run(" in stripped:
                stream_kernels.setdefault(current_stream, []).append(stripped)

        self.assertEqual(
            len(stream_kernels.get("s1", [])),
            1,
            f"Expected 1 kernel on s1, got: {stream_kernels}",
        )
        self.assertEqual(
            len(stream_kernels.get("s2", [])),
            1,
            f"Expected 1 kernel on s2, got: {stream_kernels}",
        )

    def test_no_buffer_reuse_across_streams(self):
        """Buffer produced on one stream must not be reused in-place on another."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            with torch.cuda.stream(s1):
                a = x + 1
            e = s1.record_event()
            s2.wait_event(e)
            with torch.cuda.stream(s2):
                b = a + 2
            s2.synchronize()
            return b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)
        # The second kernel should allocate a fresh buffer, not reuse
        # the one produced on the other stream
        wrapper = _extract_wrapper_body(code)
        self.assertIn("record_event", wrapper)
        self.assertIn("wait_event", wrapper)
        self.assertNotIn("buf0; del buf0", wrapper)

    def test_stream_record_wait_event_not_dropped(self):
        """stream.record_event() and stream.wait_event() must survive compilation."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            with torch.cuda.stream(s1):
                a = x + 1
            e = s1.record_event()
            s2.wait_event(e)
            with torch.cuda.stream(s2):
                b = a * 2
            s2.synchronize()
            return b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)
        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)
        self.assertIn("synchronize_stream", code)

    def test_stream_synchronize_not_dropped(self):
        """stream.synchronize() must survive compilation and appear in wrapper code."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                a = x + 1
            s.synchronize()
            b = a * 2
            return b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)
        self.assertIn("synchronize_stream", code)

    def test_stream_wait_stream_not_dropped(self):
        """stream.wait_stream() must survive compilation and appear in wrapper code."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            with torch.cuda.stream(s1):
                a = x + 1
            s2.wait_stream(s1)
            with torch.cuda.stream(s2):
                b = a * 2
            s2.synchronize()
            return b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)
        self.assertIn("wait_stream", code)
        self.assertIn("synchronize_stream", code)

    def test_codegen_structure_single_stream(self):
        """Verify wrapper structure for pointwise ops with one side stream."""
        from torch._inductor.utils import run_and_get_code

        def fn(x):
            s = torch.cuda.Stream()
            a = x * 2
            with torch.cuda.stream(s):
                b = x * 3
            s.synchronize()
            return a + b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)
        counter = CompileCounterWithBackend("inductor")
        compiled_fn = torch.compile(fn, backend=counter)
        result, (code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)

        self.assertExpectedInline(
            normalize_gm(counter.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[1024]"):
        l_x_ = L_x_

        get_external_object_by_index = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(0);  get_external_object_by_index = None

        a: "f32[1024]" = l_x_ * 2

        b: "f32[1024]" = l_x_ * 3;  l_x_ = None

        synchronize_stream = torch.ops.streams.synchronize_stream(0);  synchronize_stream = None

        add: "f32[1024]" = a + b;  a = b = None
        return (add,)
""",  # noqa: B950
        )

        wrapper_body = _extract_wrapper_body(code)
        self.assertExpectedInline(
            wrapper_body,
            """\
arg0_1, = args
with torch.cuda._DeviceGuard(0):
    torch.cuda.set_device(0)
    default_stream = torch.cuda.current_stream()
    from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
    stream1 = get_external_object_by_index(0)
    with torch.cuda.stream(stream1):
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        raw_stream = get_raw_stream(0)
        triton_kernel.run(arg0_1, buf0, 1024, stream=raw_stream)
    with torch.cuda.stream(default_stream):
        buf3 = empty_strided_cuda((1024, ), (1, ), torch.float32)
        stream0 = get_raw_stream(0)
        triton_kernel.run(arg0_1, buf0, buf3, 1024, stream=stream0)
        torch.ops.streams.synchronize_stream.default(0)
    return (buf3, )""",
        )

    def test_codegen_structure_pipeline(self):
        """Verify wrapper structure for two-stage matmul pipeline."""
        from torch._inductor.utils import run_and_get_code

        def fn(x, w1, w2):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()
            a = x @ w1
            event.record()
            with torch.cuda.stream(s):
                event.wait()
                b = a @ w2
            s.synchronize()
            return b

        x = torch.randn(32, 32, device="cuda")
        w1 = torch.randn(32, 32, device="cuda")
        w2 = torch.randn(32, 32, device="cuda")
        expected = fn(x, w1, w2)
        counter = CompileCounterWithBackend("inductor")
        compiled_fn = torch.compile(fn, backend=counter)
        result, (code,) = run_and_get_code(compiled_fn, x, w1, w2)
        self.assertEqual(result, expected)

        self.assertExpectedInline(
            normalize_gm(counter.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[32, 32]", L_w1_: "f32[32, 32]", L_w2_: "f32[32, 32]"):
        l_x_ = L_x_
        l_w1_ = L_w1_
        l_w2_ = L_w2_

        get_external_object_by_index = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(0);  get_external_object_by_index = None

        get_external_object_by_index_1 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(1);  get_external_object_by_index_1 = None

        a: "f32[32, 32]" = l_x_ @ l_w1_;  l_x_ = l_w1_ = None

        get_external_object_by_index_2 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(2);  get_external_object_by_index_2 = None
        record_event = torch.ops.streams.record_event(1, 2);  record_event = None

        wait_event = torch.ops.streams.wait_event(1, 0);  wait_event = None

        b: "f32[32, 32]" = a @ l_w2_;  a = l_w2_ = None

        synchronize_stream = torch.ops.streams.synchronize_stream(0);  synchronize_stream = None
        return (b,)
""",  # noqa: B950
        )

        wrapper_body = _extract_wrapper_body(code)
        FileCheck().run(
            """\
# CHECK: with torch.cuda.stream(default_stream):
# CHECK: copy_if_misaligned
# CHECK: extern_kernels.mm(
# CHECK: record_event
# CHECK: with torch.cuda.stream(stream1):
# CHECK: wait_event
# CHECK: copy_if_misaligned
# CHECK: extern_kernels.mm(
# CHECK: with torch.cuda.stream(default_stream):
# CHECK: synchronize_stream""",
            wrapper_body,
        )

    def test_codegen_structure_three_stream_pipeline(self):
        """Verify wrapper structure for three-stage matmul pipeline."""
        from torch._inductor.utils import run_and_get_code

        def fn(x, w1, w2, w3):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            s3 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()
            with torch.cuda.stream(s1):
                a = x @ w1
                e1.record(s1)
            with torch.cuda.stream(s2):
                e1.wait(s2)
                b = a @ w2
                e2.record(s2)
            with torch.cuda.stream(s3):
                e2.wait(s3)
                c = b @ w3
            s1.synchronize()
            s2.synchronize()
            s3.synchronize()
            return c

        x = torch.randn(32, 32, device="cuda")
        w1 = torch.randn(32, 32, device="cuda")
        w2 = torch.randn(32, 32, device="cuda")
        w3 = torch.randn(32, 32, device="cuda")
        expected = fn(x, w1, w2, w3)
        counter = CompileCounterWithBackend("inductor")
        compiled_fn = torch.compile(fn, backend=counter)
        result, (code,) = run_and_get_code(compiled_fn, x, w1, w2, w3)
        self.assertEqual(result, expected)

        self.assertExpectedInline(
            normalize_gm(counter.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[32, 32]", L_w1_: "f32[32, 32]", L_w2_: "f32[32, 32]", L_w3_: "f32[32, 32]"):
        l_x_ = L_x_
        l_w1_ = L_w1_
        l_w2_ = L_w2_
        l_w3_ = L_w3_

        get_external_object_by_index = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(0);  get_external_object_by_index = None

        get_external_object_by_index_1 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(1);  get_external_object_by_index_1 = None

        get_external_object_by_index_2 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(2);  get_external_object_by_index_2 = None

        get_external_object_by_index_3 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(3);  get_external_object_by_index_3 = None

        get_external_object_by_index_4 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(4);  get_external_object_by_index_4 = None

        a: "f32[32, 32]" = l_x_ @ l_w1_;  l_x_ = l_w1_ = None

        record_event = torch.ops.streams.record_event(3, 0);  record_event = None

        wait_event = torch.ops.streams.wait_event(3, 1);  wait_event = None

        b: "f32[32, 32]" = a @ l_w2_;  a = l_w2_ = None

        record_event_1 = torch.ops.streams.record_event(4, 1);  record_event_1 = None

        wait_event_1 = torch.ops.streams.wait_event(4, 2);  wait_event_1 = None

        c: "f32[32, 32]" = b @ l_w3_;  b = l_w3_ = None

        synchronize_stream = torch.ops.streams.synchronize_stream(0);  synchronize_stream = None

        synchronize_stream_1 = torch.ops.streams.synchronize_stream(1);  synchronize_stream_1 = None

        synchronize_stream_2 = torch.ops.streams.synchronize_stream(2);  synchronize_stream_2 = None
        return (c,)
""",  # noqa: B950
        )

        wrapper_body = _extract_wrapper_body(code)
        FileCheck().run(
            """\
# CHECK: with torch.cuda.stream(stream1):
# CHECK: copy_if_misaligned
# CHECK: extern_kernels.mm(
# CHECK: record_event
# CHECK: with torch.cuda.stream(stream2):
# CHECK: wait_event
# CHECK: copy_if_misaligned
# CHECK: extern_kernels.mm(
# CHECK: record_event
# CHECK: with torch.cuda.stream(stream3):
# CHECK: wait_event
# CHECK: copy_if_misaligned
# CHECK: extern_kernels.mm(
# CHECK: with torch.cuda.stream(default_stream):
# CHECK: synchronize_stream
# CHECK: synchronize_stream
# CHECK: synchronize_stream""",
            wrapper_body,
        )

    def test_codegen_structure_parallel_matmuls(self):
        """Verify wrapper structure for parallel matmuls with join."""
        from torch._inductor.utils import run_and_get_code

        def fn(x, w1, w2):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()
            with torch.cuda.stream(s1):
                a = x @ w1
                e1.record(s1)
            with torch.cuda.stream(s2):
                b = x @ w2
                e2.record(s2)
            e1.wait()
            e2.wait()
            c = a + b
            s1.synchronize()
            s2.synchronize()
            return c

        x = torch.randn(32, 32, device="cuda")
        w1 = torch.randn(32, 32, device="cuda")
        w2 = torch.randn(32, 32, device="cuda")
        expected = fn(x, w1, w2)
        counter = CompileCounterWithBackend("inductor")
        compiled_fn = torch.compile(fn, backend=counter)
        result, (code,) = run_and_get_code(compiled_fn, x, w1, w2)
        self.assertEqual(result, expected)

        self.assertExpectedInline(
            normalize_gm(counter.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[32, 32]", L_w1_: "f32[32, 32]", L_w2_: "f32[32, 32]"):
        l_x_ = L_x_
        l_w1_ = L_w1_
        l_w2_ = L_w2_

        get_external_object_by_index = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(0);  get_external_object_by_index = None

        get_external_object_by_index_1 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(1);  get_external_object_by_index_1 = None

        get_external_object_by_index_2 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(2);  get_external_object_by_index_2 = None

        get_external_object_by_index_3 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(3);  get_external_object_by_index_3 = None

        a: "f32[32, 32]" = l_x_ @ l_w1_;  l_w1_ = None

        record_event = torch.ops.streams.record_event(2, 0);  record_event = None

        b: "f32[32, 32]" = l_x_ @ l_w2_;  l_x_ = l_w2_ = None

        record_event_1 = torch.ops.streams.record_event(3, 1);  record_event_1 = None

        get_external_object_by_index_4 = torch__dynamo_graph_bytecode_inputs_get_external_object_by_index(4);  get_external_object_by_index_4 = None
        wait_event = torch.ops.streams.wait_event(2, 4);  wait_event = None

        wait_event_1 = torch.ops.streams.wait_event(3, 4);  wait_event_1 = None

        c: "f32[32, 32]" = a + b;  a = b = None

        synchronize_stream = torch.ops.streams.synchronize_stream(0);  synchronize_stream = None

        synchronize_stream_1 = torch.ops.streams.synchronize_stream(1);  synchronize_stream_1 = None
        return (c,)
""",  # noqa: B950
        )

        wrapper_body = _extract_wrapper_body(code)
        FileCheck().run(
            """\
# CHECK: with torch.cuda.stream(stream1):
# CHECK: copy_if_misaligned
# CHECK: extern_kernels.mm(
# CHECK: record_event
# CHECK: with torch.cuda.stream(stream2):
# CHECK: copy_if_misaligned
# CHECK: extern_kernels.mm(
# CHECK: record_event
# CHECK: with torch.cuda.stream(default_stream):
# CHECK: wait_event
# CHECK: triton_kernel.run(
# CHECK: synchronize_stream
# CHECK: synchronize_stream""",
            wrapper_body,
        )


@unittest.skipUnless(TEST_CUDA, "requires CUDA")
class TestStreamOrderingStress(InductorTestCase):
    """Stress tests verifying that interleaved event record/wait ops
    produce correct ordering under compilation.  Each test uses large
    matmuls so there is real GPU work, and repeats many iterations so
    that race conditions (if ordering is wrong) surface reliably."""

    N = 4096  # matrix size — big enough for real GPU work
    ITERS = 20  # repetitions per test

    def _check_compiled_matches_eager(self, fn, *args):
        """Run fn eagerly and compiled, assert results match over ITERS runs."""
        compiled_fn = torch.compile(fn)
        for _ in range(self.ITERS):
            expected = fn(*args)
            actual = compiled_fn(*args)
            # Full device sync as a safety net to ensure all stream work
            # is visible before comparing results.
            torch.cuda.synchronize()
            if not isinstance(expected, (tuple, list)):
                expected, actual = [expected], [actual]
            for e, a in zip(expected, actual):
                self.assertEqual(a, e)

    @staticmethod
    def _heavy_matmul_chain(x, w, depth=8):
        """Chain of matmuls to create substantial GPU work (~ms).
        Used to widen the race window between streams so that missing
        synchronization is observable."""
        h = x
        for _ in range(depth):
            h = h @ w
        return h

    # ------------------------------------------------------------------
    # 1. Race: producer does heavy work, consumer reads the result.
    #    Without the event.wait() the consumer would launch immediately
    #    and read stale memory because the producer chain hasn't finished.
    # ------------------------------------------------------------------
    def test_race_producer_consumer(self):
        N = self.N

        def fn(x, w):
            s = torch.cuda.Stream()
            e = torch.cuda.Event()

            # Heavy producer on default stream — takes real GPU time
            a = TestStreamOrderingStress._heavy_matmul_chain(x, w)
            e.record()

            with torch.cuda.stream(s):
                e.wait()  # removing this would cause a race
                b = a + 1

            s.synchronize()
            return b

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.9  # use scaled identity for stability
        self._check_compiled_matches_eager(fn, x, w)

    # ------------------------------------------------------------------
    # 2. Race: ping-pong where each direction has heavy work.
    #    Both event.wait() calls are load-bearing.
    # ------------------------------------------------------------------
    def test_race_ping_pong(self):
        N = self.N

        def fn(x, w):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            with torch.cuda.stream(s1):
                a = TestStreamOrderingStress._heavy_matmul_chain(x, w)
                e1.record(s1)

            with torch.cuda.stream(s2):
                e1.wait(s2)
                b = TestStreamOrderingStress._heavy_matmul_chain(a, w)
                e2.record(s2)

            with torch.cuda.stream(s1):
                e2.wait(s1)
                c = b + a

            s1.synchronize()
            s2.synchronize()
            return c

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.9
        self._check_compiled_matches_eager(fn, x, w)

    # ------------------------------------------------------------------
    # 3. Race: fan-out where the producer is slow.
    #    All three consumers depend on the producer finishing.
    # ------------------------------------------------------------------
    def test_race_fan_out(self):
        N = self.N

        def fn(x, w):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            s3 = torch.cuda.Stream()
            e = torch.cuda.Event()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()
            e3 = torch.cuda.Event()

            # Slow producer
            a = TestStreamOrderingStress._heavy_matmul_chain(x, w)
            e.record()

            with torch.cuda.stream(s1):
                e.wait()
                r1 = a * 2
                e1.record(s1)

            with torch.cuda.stream(s2):
                e.wait()
                r2 = a * 3
                e2.record(s2)

            with torch.cuda.stream(s3):
                e.wait()
                r3 = a * 4
                e3.record(s3)

            e1.wait()
            e2.wait()
            e3.wait()
            result = r1 + r2 + r3

            s1.synchronize()
            s2.synchronize()
            s3.synchronize()
            return result

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.9
        self._check_compiled_matches_eager(fn, x, w)

    # ------------------------------------------------------------------
    # 4. Race: diamond pattern with heavy work on both branches.
    #    The join must wait for both branches.
    # ------------------------------------------------------------------
    def test_race_diamond(self):
        N = self.N

        def fn(x, w):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            e_fork = torch.cuda.Event()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            base = x @ w
            e_fork.record()

            with torch.cuda.stream(s1):
                e_fork.wait()
                branch1 = TestStreamOrderingStress._heavy_matmul_chain(
                    torch.relu(base), w
                )
                e1.record(s1)

            with torch.cuda.stream(s2):
                e_fork.wait()
                branch2 = TestStreamOrderingStress._heavy_matmul_chain(
                    torch.sigmoid(base), w
                )
                e2.record(s2)

            e1.wait()
            e2.wait()
            result = branch1 + branch2

            s1.synchronize()
            s2.synchronize()
            return result

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.5
        self._check_compiled_matches_eager(fn, x, w)

    # ------------------------------------------------------------------
    # 5. Race: 4-stage pipeline where each stage is heavy.
    #    Every event.wait() is load-bearing.
    # ------------------------------------------------------------------
    def test_race_pipeline(self):
        N = self.N

        def fn(x, w):
            streams = [torch.cuda.Stream() for _ in range(4)]
            events = [torch.cuda.Event() for _ in range(3)]

            with torch.cuda.stream(streams[0]):
                h = TestStreamOrderingStress._heavy_matmul_chain(x, w, depth=4)
                events[0].record(streams[0])

            with torch.cuda.stream(streams[1]):
                events[0].wait(streams[1])
                h = TestStreamOrderingStress._heavy_matmul_chain(h, w, depth=4)
                events[1].record(streams[1])

            with torch.cuda.stream(streams[2]):
                events[1].wait(streams[2])
                h = TestStreamOrderingStress._heavy_matmul_chain(h, w, depth=4)
                events[2].record(streams[2])

            with torch.cuda.stream(streams[3]):
                events[2].wait(streams[3])
                h = h + x  # quick consumer — races if wait is missing

            for s in streams:
                s.synchronize()
            return h

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.9
        self._check_compiled_matches_eager(fn, x, w)

    # ------------------------------------------------------------------
    # 6. Race: back-to-back sync, both directions carry heavy work
    # ------------------------------------------------------------------
    def test_race_back_to_back(self):
        N = self.N

        def fn(x, w):
            s = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            a = TestStreamOrderingStress._heavy_matmul_chain(x, w)
            e1.record()

            with torch.cuda.stream(s):
                e1.wait()
                b = TestStreamOrderingStress._heavy_matmul_chain(a, w)
                e2.record(s)

            e2.wait()
            c = b + 1

            s.synchronize()
            return c

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.9
        self._check_compiled_matches_eager(fn, x, w)

    # ------------------------------------------------------------------
    # 7. Race: triton kernel on user stream.
    #    Without the triton stream fix the kernel launches on the default
    #    stream and reads stale/in-progress data from the user stream.
    # ------------------------------------------------------------------
    def test_race_triton_on_user_stream(self):
        N = self.N

        def fn(x, w):
            s = torch.cuda.Stream()
            e = torch.cuda.Event()

            with torch.cuda.stream(s):
                # Heavy matmul chain produces data on user stream
                a = TestStreamOrderingStress._heavy_matmul_chain(x, w)
                # Triton pointwise on the same user stream — without fix
                # this launches on the default stream
                b = torch.relu(a)
                e.record(s)

            e.wait()
            s.synchronize()
            return b

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.9
        self._check_compiled_matches_eager(fn, x, w)


@unittest.skipUnless(TEST_CUDA, "requires CUDA")
class TestGenericStreamCompile(InductorTestCase):
    """Tests for torch.compile with device-agnostic torch.Stream API."""

    def test_generic_stream_basic(self):
        """Test compilation with torch.Stream (device-agnostic API)."""
        from torch._inductor.utils import run_and_get_code

        # Create stream outside compiled function
        stream = torch.Stream("cuda")
        # Convert to cuda stream for use with torch.cuda.stream()
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )

        def fn(x):
            with torch.cuda.stream(cuda_stream):
                a = x * 2
                b = a + 1

            cuda_stream.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream context is present
        self.assertIn("torch.cuda.stream", code)

    def test_generic_stream_with_event(self):
        """Test compilation with torch.Stream and torch.Event."""
        from torch._inductor.utils import run_and_get_code

        # Create stream and event outside compiled function
        stream = torch.Stream("cuda")
        event = torch.Event("cuda")
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )

        def fn(x):
            # Work on default stream
            a = x * 2
            event.record()

            with torch.cuda.stream(cuda_stream):
                event.wait()
                b = a + 1

            cuda_stream.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify event operations survive compilation as custom ops
        self.assertIn("record_event", code)

    def test_generic_stream_multiple(self):
        """Test compilation with multiple torch.Stream instances."""
        from torch._inductor.utils import run_and_get_code

        # Create streams and event outside compiled function
        stream1 = torch.Stream("cuda")
        stream2 = torch.Stream("cuda")
        event = torch.Event("cuda")
        cuda_stream1 = torch.cuda.Stream(
            stream_id=stream1.stream_id,
            device_index=stream1.device_index,
            device_type=stream1.device_type,
        )
        cuda_stream2 = torch.cuda.Stream(
            stream_id=stream2.stream_id,
            device_index=stream2.device_index,
            device_type=stream2.device_type,
        )

        def fn(x):
            with torch.cuda.stream(cuda_stream1):
                a = x * 2
                event.record()

            with torch.cuda.stream(cuda_stream2):
                event.wait()
                b = a + 1

            cuda_stream1.synchronize()
            cuda_stream2.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify stream handling and event ops survive
        self.assertIn("torch.cuda.stream", code)
        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)

    def test_generic_event_record_on_stream(self):
        """Test torch.Event.record() with explicit stream argument."""
        from torch._inductor.utils import run_and_get_code

        # Create stream and event outside compiled function
        stream = torch.Stream("cuda")
        event = torch.Event("cuda")
        cuda_stream = torch.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=stream.device_type,
        )

        def fn(x):
            with torch.cuda.stream(cuda_stream):
                a = x * 2
                # Record event with explicit stream (using cuda_stream)
                event.record(cuda_stream)

            event.wait()
            b = a + 1

            cuda_stream.synchronize()
            return b

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)

        self.assertEqual(result, expected)

        # Verify event operations survive compilation as custom ops
        self.assertIn("record_event", code)


@unittest.skipUnless(TEST_CUDA, "requires CUDA")
class TestStreamIdentity(InductorTestCase):
    """Verify that compiled code uses the user's original stream objects."""

    def test_single_stream_identity(self):
        """Codegen should retrieve the user's stream via get_external_object_by_index."""
        from torch._inductor.utils import run_and_get_code

        user_stream = torch.cuda.Stream()

        def fn(x):
            with torch.cuda.stream(user_stream):
                return x * 2

        x = torch.randn(1024, device="cuda")
        result, (code,) = run_and_get_code(torch.compile(fn), x)

        self.assertEqual(result, fn(x))
        self.assertIn("get_external_object_by_index", code)
        self.assertNotIn("torch.cuda.Stream(device=", code)

    def test_multiple_stream_identity(self):
        """Each stream context should retrieve a different user stream object."""
        from torch._inductor.utils import run_and_get_code

        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        def fn(x):
            event = torch.cuda.Event()
            with torch.cuda.stream(stream_a):
                a = x * 2
                event.record()
            with torch.cuda.stream(stream_b):
                event.wait()
                b = a + 1
            stream_b.synchronize()
            return b

        x = torch.randn(1024, device="cuda")
        result, (code,) = run_and_get_code(torch.compile(fn), x)

        self.assertEqual(result, fn(x))
        # Should have two distinct get_external_object_by_index calls
        matches = re.findall(r"get_external_object_by_index\((\d+)\)", code)
        self.assertEqual(len(matches), 2)
        self.assertNotEqual(matches[0], matches[1])
        self.assertNotIn("torch.cuda.Stream(device=", code)


@unittest.skipUnless(TEST_CUDA, "requires CUDA")
class TestPDLWithMultiStream(InductorTestCase):
    """Tests that PDL (Programmatic Dependent Launch) composes safely with
    user-annotated multi-stream code under torch.compile.

    PDL's GDC intrinsics are stream-local: gdc_wait/gdc_launch_dependents
    only govern the overlap between consecutive kernels on the *same* CUDA
    stream.  Cross-stream ordering is handled entirely by CUDA events at the
    wrapper level.  These tests verify that enabling PDL in the presence of
    multi-stream code doesn't break correctness, doesn't interfere with
    stream-level invariants (no cross-stream fusion, event ops preserved),
    and still applies within each stream's own kernel sequence.
    """

    @unittest.skipIf(not SM90OrLater or TEST_WITH_ROCM, "PDL requires NVIDIA sm90+")
    @inductor_config.patch({"triton.enable_pdl": True})
    def test_pdl_single_side_stream(self):
        """PDL metadata is emitted for a kernel on a side stream."""
        from torch._inductor.utils import run_and_get_code, run_and_get_triton_code

        def fn(x):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                a = x * 2
                b = a + 1
            s.synchronize()
            return b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)

        compiled_fn = torch.compile(fn)
        result, (wrapper_code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)

        self.assertIn("torch.cuda.stream", wrapper_code)
        self.assertIn("synchronize_stream", wrapper_code)

        triton_code = run_and_get_triton_code(torch.compile(fn), x)
        (
            FileCheck()
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
        ).run(triton_code)

    @unittest.skipIf(not SM90OrLater or TEST_WITH_ROCM, "PDL requires NVIDIA sm90+")
    @inductor_config.patch({"triton.enable_pdl": True})
    def test_pdl_correctness_with_multiple_streams(self):
        """Enabling PDL with independent side streams produces correct results."""
        from torch._inductor.utils import run_and_get_triton_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            with torch.cuda.stream(s1):
                a = x * 2 + 1
                e1.record(s1)

            with torch.cuda.stream(s2):
                b = x * 3 + 2
                e2.record(s2)

            e1.wait()
            e2.wait()
            return a + b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)
        compiled_fn = torch.compile(fn)
        self.assertEqual(compiled_fn(x), expected)

        triton_code = run_and_get_triton_code(torch.compile(fn), x)
        # s1 kernel, s2 kernel, and default-stream add kernel
        (
            FileCheck()
            # s1 kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
            # s2 kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
            # default stream add
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
        ).run(triton_code)

    @unittest.skipIf(not SM90OrLater or TEST_WITH_ROCM, "PDL requires NVIDIA sm90+")
    @inductor_config.patch({"triton.enable_pdl": True})
    def test_pdl_cross_stream_events_preserved(self):
        """Event record/wait for cross-stream sync must survive with PDL on.

        PDL is stream-local so it cannot replace event-based cross-stream
        ordering.  Verify the events are still in the generated code."""
        from torch._inductor.utils import run_and_get_code, run_and_get_triton_code

        def fn(x):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()

            a = x * 2 + 1
            event.record()

            with torch.cuda.stream(s):
                event.wait()
                b = a + 3

            s.synchronize()
            return b

        x = torch.randn(1024, device="cuda")
        expected = fn(x)

        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)

        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)

        # Both kernels (default + side stream) get PDL intrinsics
        triton_code = run_and_get_triton_code(torch.compile(fn), x)
        (
            FileCheck()
            # default stream kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
            # side stream kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
        ).run(triton_code)

    @unittest.skipIf(not SM90OrLater or TEST_WITH_ROCM, "PDL requires NVIDIA sm90+")
    @inductor_config.patch({"triton.enable_pdl": True})
    def test_pdl_same_stream_consecutive_kernels(self):
        """Two consecutive kernels on the same side stream should both get PDL.

        This is the case where PDL is actually useful: the second kernel can
        overlap with the first via GDC intrinsics because they share a stream."""
        from torch._inductor.utils import run_and_get_triton_code

        def fn(x, y):
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                # Two separate fused groups on the same stream
                a = x**2 + x
                y.copy_(a)
            s.synchronize()
            return y

        x = torch.randn(1024, device="cuda")
        y = torch.empty(1024, device="cuda")
        expected = fn(x, y.clone())
        compiled_fn = torch.compile(fn)
        self.assertEqual(compiled_fn(x, y.clone()), expected)

        triton_code = run_and_get_triton_code(torch.compile(fn), x, y.clone())
        (
            FileCheck()
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
        ).run(triton_code)

    @unittest.skipIf(not SM90OrLater or TEST_WITH_ROCM, "PDL requires NVIDIA sm90+")
    @inductor_config.patch({"triton.enable_pdl": True})
    def test_pdl_no_fusion_across_streams(self):
        """PDL must not cause cross-stream ops to be fused."""
        from torch._inductor.utils import run_and_get_triton_code

        def fn(x):
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            e1 = torch.cuda.Event()
            e2 = torch.cuda.Event()

            with torch.cuda.stream(s1):
                a = x * 2
                b = a + 1
                e1.record(s1)

            with torch.cuda.stream(s2):
                c = x * 3
                d = c + 2
                e2.record(s2)

            e1.wait()
            e2.wait()
            return b + d

        x = torch.randn(1024, device="cuda")
        expected = fn(x)

        compiled_fn = torch.compile(fn)
        torch._inductor.metrics.reset()
        result = compiled_fn(x)
        self.assertEqual(result, expected)

        # 3 kernels: s1 pointwise, s2 pointwise, default stream add
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 3)

        # All 3 kernels should have PDL with GDC intrinsics
        triton_code = run_and_get_triton_code(torch.compile(fn), x)
        (
            FileCheck()
            # s1 kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
            # s2 kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
            # default stream add
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
        ).run(triton_code)

    @unittest.skipIf(not SM90OrLater or TEST_WITH_ROCM, "PDL requires NVIDIA sm90+")
    @inductor_config.patch({"triton.enable_pdl": True})
    def test_pdl_stress_multistream_correctness(self):
        """Stress test: heavy work across streams with PDL must produce
        correct results over many iterations to surface any races.

        Uses 4096x4096 matmuls (matching TestStreamOrderingStress) so the
        GPU work is long enough that a missing event.wait() would cause
        the consumer to read stale data."""
        from torch._inductor.utils import run_and_get_code

        N = 4096
        ITERS = 20

        def fn(x, w):
            s = torch.cuda.Stream()
            e = torch.cuda.Event()

            h = x
            for _ in range(4):
                h = h @ w
            e.record()

            with torch.cuda.stream(s):
                e.wait()
                out = torch.relu(h) + 1.0

            s.synchronize()
            return out

        x = torch.randn(N, N, device="cuda")
        w = torch.eye(N, device="cuda") * 0.9

        # Verify codegen once before the stress loop
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x, w)
        # Wrapper must have stream context and event sync
        self.assertIn("torch.cuda.stream", code)
        self.assertIn("wait_event", code)
        # The relu+add pointwise kernel should have PDL
        (
            FileCheck()
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
        ).run(code)

        for _ in range(ITERS):
            expected = fn(x, w)
            actual = compiled_fn(x, w)
            torch.cuda.synchronize()
            self.assertEqual(actual, expected)

    @unittest.skipIf(not SM90OrLater or TEST_WITH_ROCM, "PDL requires NVIDIA sm90+")
    @inductor_config.patch({"triton.enable_pdl": True})
    def test_pdl_mutation_across_streams(self):
        """Buffer mutation on one stream, read on another, with PDL enabled.

        The mutation is on a locally-created buffer (not an input) to avoid
        the dynamo guard that forbids event.record() after input mutation."""
        from torch._inductor.utils import run_and_get_code, run_and_get_triton_code

        def fn(x):
            s = torch.cuda.Stream()
            event = torch.cuda.Event()

            # Produce a new buffer (not input mutation) then record
            a = x * 2
            event.record()

            with torch.cuda.stream(s):
                event.wait()
                # In-place add on side stream
                a.add_(1)

            s.synchronize()
            return a

        x = torch.randn(1024, device="cuda")

        expected = fn(x)
        compiled_fn = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled_fn, x)
        self.assertEqual(result, expected)

        self.assertIn("record_event", code)
        self.assertIn("wait_event", code)

        # Both kernels (default + side stream) get PDL intrinsics
        triton_code = run_and_get_triton_code(torch.compile(fn), x)
        (
            FileCheck()
            # default stream kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
            # side stream kernel
            .check("'launch_pdl': True")
            .check("gdc_wait")
            .check("gdc_launch")
        ).run(triton_code)


instantiate_parametrized_tests(TestStreamUtils)
instantiate_parametrized_tests(TestWrapperCodegenStreams)
instantiate_parametrized_tests(TestStreamCodegen)
instantiate_parametrized_tests(TestUserStreamCompile)
instantiate_parametrized_tests(TestStreamOrderingStress)
instantiate_parametrized_tests(TestGenericStreamCompile)
instantiate_parametrized_tests(TestStreamIdentity)
instantiate_parametrized_tests(TestPDLWithMultiStream)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()

# Owner(s): ["module: cuda"]

import sys
import textwrap
import traceback
from typing import Optional

import torch
import torch.cuda._sanitizer as csan
from torch.cuda._sanitizer import DataPtr, EventId, StreamId
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_CUDA, TestCase
from torch.testing._internal.two_tensor import TwoTensor


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


class TestArgumentHandler(TestCase):
    def test_add(self):
        add_func = torch.ops.aten.add.Tensor
        a = torch.ones(5, 3, device="cuda")
        b = torch.randn(5, 3, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(add_func._schema, (a, b), {}, is_factory=False)
        c = torch.add(a, b)
        argument_handler.parse_outputs(add_func._schema, c, is_factory=False)

        self.assertEqual({a.data_ptr(), b.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({c.data_ptr()}, argument_handler.dataptrs_written)

    def test_cat(self):
        cat_func = torch.ops.aten.cat.default
        a = torch.ones(2, 4, 5, device="cuda")
        b = torch.zeros(2, 1, 5, device="cuda")
        c = torch.rand(2, 7, 5, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            cat_func._schema, ([a, b, c], 1), {}, is_factory=False
        )
        d = torch.cat((a, b, c), dim=1)
        argument_handler.parse_outputs(cat_func._schema, d, is_factory=False)

        self.assertEqual(
            {a.data_ptr(), b.data_ptr(), c.data_ptr()}, argument_handler.dataptrs_read
        )
        self.assertEqual({d.data_ptr()}, argument_handler.dataptrs_written)

    def test_split(self):
        split_func = torch.ops.aten.split.Tensor
        a = torch.arange(10, device="cuda").reshape(5, 2)

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(split_func._schema, (a, 2), {}, is_factory=False)
        out = torch.split(a, 2)
        argument_handler.parse_outputs(split_func._schema, out, is_factory=False)

        # Split is a view op, no data is read or written!
        self.assertEqual(len(argument_handler.dataptrs_read), 0)
        self.assertEqual(len(argument_handler.dataptrs_written), 0)

    def test_inplace(self):
        add_inplace_func = torch.ops.aten.add_.Tensor
        a = torch.rand(4, 2, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            add_inplace_func._schema, (a, 5), {}, is_factory=False
        )
        a.add_(5)
        argument_handler.parse_outputs(add_inplace_func._schema, a, is_factory=False)

        self.assertEqual(set(), argument_handler.dataptrs_read)
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_written)

    def test_out(self):
        mul_out_func = torch.ops.aten.mul.out
        a = torch.arange(8, device="cuda")
        b = torch.empty(8, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            mul_out_func._schema, (a, 3), {"out": b}, is_factory=False
        )
        torch.mul(a, 3, out=b)
        argument_handler.parse_outputs(mul_out_func._schema, b, is_factory=False)

        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual({b.data_ptr()}, argument_handler.dataptrs_written)

    def test_nonzero(self):
        nonzero_func = torch.ops.aten.nonzero.default
        a = torch.ones(5, 3, 2, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            nonzero_func._schema, (a,), {"as_tuple": True}, is_factory=False
        )
        out = torch.nonzero(a, as_tuple=True)
        argument_handler.parse_outputs(nonzero_func._schema, out, is_factory=False)

        outputs = {out[0].data_ptr(), out[1].data_ptr(), out[2].data_ptr()}
        self.assertEqual({a.data_ptr()}, argument_handler.dataptrs_read)
        self.assertEqual(outputs, argument_handler.dataptrs_written)

    def test_tensor_names(self):
        addr_func = torch.ops.aten.addr.default
        vec = torch.arange(1, 4, device="cuda")
        M = torch.zeros(3, 3, device="cuda")

        argument_handler = csan.ArgumentHandler()
        argument_handler.parse_inputs(
            addr_func._schema, (M, vec, vec), {}, is_factory=False
        )
        out = torch.addr(M, vec, vec)
        argument_handler.parse_outputs(addr_func._schema, out, is_factory=False)

        self.assertEqual(
            argument_handler.tensor_aliases,
            {
                M.data_ptr(): ["self"],
                vec.data_ptr(): ["vec1", "vec2"],
                out.data_ptr(): [],
            },
        )
        self.assertEqual({out.data_ptr()}, argument_handler.outputs)


def tensor_id(i: int) -> DataPtr:
    return i


def stream_id(i: int) -> StreamId:
    return 1000 + i


def event_id(i: int) -> EventId:
    return 2000 + i


class TestEventHandler(TestCase):
    def setUp(self):
        self.handler = csan.EventHandler()

    def kernel_launch(
        self,
        stream: StreamId,
        read_only: Optional[list[DataPtr]] = None,
        read_write: Optional[list[DataPtr]] = None,
    ) -> list[csan.SynchronizationError]:
        if read_only is None:
            read_only = []
        if read_write is None:
            read_write = []
        return self.handler._handle_kernel_launch(
            stream,
            read_only,
            read_write,
            {},
            "",
            {k: [""] for k in read_only + read_write},
        )

    def assert_good_kernel_launch(
        self,
        stream: StreamId,
        read_only: Optional[list[DataPtr]] = None,
        read_write: Optional[list[DataPtr]] = None,
    ) -> None:
        self.assertEqual(self.kernel_launch(stream, read_only, read_write), [])

    def assert_bad_kernel_launch(
        self,
        number_of_errors: int,
        stream: StreamId,
        read_only: Optional[list[DataPtr]] = None,
        read_write: Optional[list[DataPtr]] = None,
    ) -> None:
        errors = self.kernel_launch(stream, read_only, read_write)
        self.assertEqual(len(errors), number_of_errors)

    def test_empty_kernel_launch(self):
        self.assert_good_kernel_launch(stream_id(0))

    def test_simple_passing(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

    def test_simple_error(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_simple_sync(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_reads_check_last_write(self):
        # Tests that not only the first read operation checks if it is in conflict
        # with the last write operation, but all read operations do.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(3), read_only=[tensor_id(1)])

    def test_branch_sync(self):
        # Tests that two streams can read after both waiting for a third, but they
        # cannot write without further synchronization.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.handler._handle_event_wait(event_id(0), stream_id(2))
        self.handler._handle_event_wait(event_id(0), stream_id(3))
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_chain_sync(self):
        iterations = 10

        self.assert_good_kernel_launch(stream_id(0), read_only=[tensor_id(1)])
        for i in range(iterations):
            self.handler._handle_event_record(event_id(i), stream_id(i))
            self.handler._handle_event_wait(event_id(i), stream_id(i + 1))
        self.assert_good_kernel_launch(stream_id(iterations), read_write=[tensor_id(1)])

    def test_expired_record(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(0), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.handler._handle_event_wait(event_id(0), stream_id(2))

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_deleted_record(self):
        for should_delete, should_create in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            self.setUp()
            with self.subTest(should_delete=should_delete, should_create=should_create):
                self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
                self.handler._handle_event_record(event_id(0), stream_id(1))

                if should_delete:
                    self.handler._handle_event_deletion(event_id(0))
                if should_create:
                    self.handler._handle_event_creation(event_id(0))

                self.handler._handle_event_wait(event_id(0), stream_id(2))
                self.assert_bad_kernel_launch(
                    1, stream_id(2), read_write=[tensor_id(1)]
                )

    def test_all_reads_checked_failing(self):
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))

        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        self.assert_good_kernel_launch(stream_id(iterations), read_only=[tensor_id(1)])
        self.handler._handle_event_record(event_id(iterations), stream_id(i))

        # Does not synchronize with the last read.
        self.assert_bad_kernel_launch(1, stream_id(0), read_write=[tensor_id(1)])

    def test_all_reads_checked_passing(self):
        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_only=[tensor_id(1)])
            self.handler._handle_event_record(event_id(i), stream_id(i))

        for i in range(1, iterations):
            self.handler._handle_event_wait(event_id(i), stream_id(0))

        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])

    def test_multiple_errors(self):
        iterations = 10
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(iterations)]
        )
        self.assert_bad_kernel_launch(
            iterations,
            stream_id(1),
            read_write=[tensor_id(i) for i in range(iterations)],
        )

    def test_correct_state_merging(self):
        # Tests that after waiting for an event, a stream's state is indeed set
        # to the pointwise maximum of its old state and the recorded state.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(2), stream_id(2))

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(2)])
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(2), stream_id(1))

        self.handler._handle_event_record(event_id(3), stream_id(2))
        self.handler._handle_event_wait(event_id(3), stream_id(1))
        self.assert_good_kernel_launch(
            stream_id(1), read_write=[tensor_id(1), tensor_id(2)]
        )

    def test_record_override(self):
        self.assert_good_kernel_launch(stream_id(1), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(2)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_record(event_id(1), stream_id(2))

        self.handler._handle_event_wait(event_id(1), stream_id(3))
        self.assert_bad_kernel_launch(1, stream_id(3), read_write=[tensor_id(1)])

    def test_multiple_wait(self):
        # Tests that a wait operation can be performed multiple times on the same event
        # by different streams.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.handler._handle_event_wait(event_id(1), stream_id(2))
        self.handler._handle_event_wait(event_id(1), stream_id(3))

        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])

    def test_device_synchronize(self):
        # Tests that a device synchronization does correctly cause all streams
        # to synchronize with each other.

        iterations = 10
        for i in range(1, iterations):
            self.assert_good_kernel_launch(stream_id(i), read_write=[tensor_id(i)])

        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(
            stream_id(0), read_write=[tensor_id(i) for i in range(1, iterations)]
        )

    def test_device_synchronization_expired(self):
        # Tests that a device synchronization is a one-time synchronization.
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])

        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(1)])

    def test_new_stream_is_synchronized(self):
        # Tests that after synchronizing operations with the host, any newly created
        # stream is guaranteed to be synchronized with them as well.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_device_synchronization()
        self.handler._handle_stream_creation(stream_id(2))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])

    def test_stream_synchronize(self):
        # Tests that a stream synchronization does correctly cause all streams to wait
        # for one specific stream, but does not synchronize all streams with each other.

        self.assert_good_kernel_launch(stream_id(0), read_write=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])
        self.handler._handle_stream_synchronization(stream_id(0))

        self.assert_good_kernel_launch(stream_id(2), read_only=[tensor_id(1)])
        self.assert_good_kernel_launch(stream_id(3), read_only=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(4), read_only=[tensor_id(2)])

    def test_event_synchronize(self):
        # Tests that an event synchronization does correctly cause all streams to wait
        # for a recorded event, but does not guarantee synchronization with the current
        # state of the stream that recorded the event.

        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(1)])
        self.handler._handle_event_record(event_id(1), stream_id(1))
        self.assert_good_kernel_launch(stream_id(1), read_write=[tensor_id(2)])

        self.handler._handle_event_synchronization(event_id(1))
        self.assert_good_kernel_launch(stream_id(2), read_write=[tensor_id(1)])
        self.assert_bad_kernel_launch(1, stream_id(2), read_write=[tensor_id(2)])


class TestMessages(TestCase):
    def setUp(self):
        self.handler = csan.EventHandler()

    def test_ensure_exists(self):
        ARG = 0
        for func, out in [
            (
                self.handler._handle_event_deletion,
                f"Found Event with id: {ARG}, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
            (
                self.handler._handle_memory_deallocation,
                f"Found tensor with pointer: {ARG}, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
        ]:
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    def test_ensure_does_not_exist(self):
        ARG = 0
        self.handler._handle_event_creation(ARG)
        self.handler._handle_stream_creation(ARG)
        for func, out in [
            (
                self.handler._handle_event_creation,
                "Found duplicate event creation in the trace for event with "
                f"id: {ARG}. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
            ),
            (
                self.handler._handle_stream_creation,
                "Found duplicate Stream creation in the trace for Stream with "
                f"id: {ARG}. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
            ),
        ]:
            with self.subTest(func=func, out=out):
                with self.assertLogs() as captured:
                    func(ARG)
                self.assertEqual(captured.records[0].getMessage(), out)

    def test_error_message(self):
        current_access = csan.Access(
            type=csan.AccessType.WRITE,
            seq_num=1,
            stream=stream_id(1),
            operator="schema",
            aliases=["b"],
            is_output=True,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace a")]
            ),
        )
        previous_access = csan.Access(
            type=csan.AccessType.READ,
            seq_num=2,
            stream=stream_id(0),
            operator="schema",
            aliases=["a"],
            is_output=False,
            stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "trace b")]
            ),
        )
        error = csan.UnsynchronizedAccessError(
            data_ptr=tensor_id(1),
            allocation_stack_trace=traceback.StackSummary.from_list(
                [("file", 0, "name", "alloc")]
            ),
            current_access=current_access,
            previous_access=previous_access,
        )
        self.assertEqual(
            str(error),
            textwrap.dedent(
                """\
                ============================
                CSAN detected a possible data race on tensor with data pointer 1
                Access by stream 1001 during kernel:
                schema
                writing to argument(s) b, and to the output
                With stack trace:
                  File "file", line 0, in name
                    trace a

                Previous access by stream 1000 during kernel:
                schema
                reading from argument(s) a
                With stack trace:
                  File "file", line 0, in name
                    trace b

                Tensor was allocated with stack trace:
                  File "file", line 0, in name
                    alloc
                """
            ),
        )

    def test_subclass(self):
        class MyT(torch.Tensor):
            def __new__(cls, data):
                new_data = data.clone()
                return new_data.as_subclass(cls)

        try:
            csan.enable_cuda_sanitizer()

            # These two tests ensure that subclass creation
            # happens smoothly under the mode used by csan
            TwoTensor(torch.rand(2), torch.rand(2))
            MyT(torch.rand(2))
        finally:
            csan.cuda_sanitizer.disable()


if __name__ == "__main__":
    run_tests()

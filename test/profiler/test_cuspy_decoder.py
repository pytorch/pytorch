# Owner(s): ["oncall: profiler"]

import ctypes
import struct
import time
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUPTI, TEST_CUPTI_V13_5
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


# The decode tests drive the real libcupti (pylibcupti); guard its import on TEST_CUPTI
# (cupti-python imports its enums at load). TEST_CUPTI_V13_5 additionally requires a loaded
# libcupti >= 13.5 on a CUDA >= 13.5 build, the Cuspy test floor.
if TEST_CUPTI:
    from torch.profiler._cuspy.cupti_python import pylibcupti


class _FieldLayout(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("fieldId", ctypes.c_int),
        ("offset", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("alignment", ctypes.c_size_t),
    ]


class _RecordLayout(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("pEntries", ctypes.POINTER(_FieldLayout)),
        ("numFields", ctypes.c_size_t),
        ("recordSize", ctypes.c_size_t),
    ]


class _CompleteInfo(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("threadId", ctypes.c_uint64),
        ("ppRecordLayouts", ctypes.POINTER(ctypes.POINTER(_RecordLayout))),
        ("numRecordLayouts", ctypes.c_size_t),
    ]


class TestCuspyDecoder(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "case",
        [
            subtest((b"", None), name="clean_end"),
            subtest((struct.pack("=I", 0x7FFFFFFF), None), name="padding"),
            subtest((struct.pack("=I", 255), None), name="unknown_kind"),
            subtest((struct.pack("=I", 10)[:1], None), name="partial_kind"),
            subtest((struct.pack("=IQ", 10, 99), None), name="partial_record"),
            subtest((b"", 0), name="zero_size"),
            subtest((b"", 3), name="short_size"),
            subtest((b"", 5), name="field_out_of_bounds"),
            subtest((b"", (1 << 64) - 1), name="oversized_record"),
        ],
    )
    def test_saved_layouts_and_tail(self, case):
        suffix, invalid_size = case
        native = torch._C._profiler._cuspy
        request = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        )(native.buffer_request_callback_address())
        complete = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p
        )(native.buffer_complete_callback_address())
        native.stop_decoder()
        native.reset_buffers()
        native.configure_buffers(4096)
        native.configure_decoder(0, 0, 10, 2)
        native.drain_decoded()
        try:
            for size, fields, values in (
                (24, [(0, 0, 4), (1, 8, 8), (2, 16, 8)], [(1, 2), (3, 4)]),
                (32, [(0, 0, 4), (1, 16, 8), (2, 24, 8)], [(5, 6)]),
            ):
                entries = (_FieldLayout * len(fields))(
                    *[
                        _FieldLayout(
                            ctypes.sizeof(_FieldLayout), field, offset, width, 1
                        )
                        for field, offset, width in fields
                    ]
                )
                layout = _RecordLayout(
                    ctypes.sizeof(_RecordLayout),
                    entries,
                    len(fields),
                    size if invalid_size is None else invalid_size,
                )
                layouts = (ctypes.POINTER(_RecordLayout) * 11)()
                layouts[10] = ctypes.pointer(layout)
                info = _CompleteInfo(
                    ctypes.sizeof(_CompleteInfo), 0, layouts, len(layouts)
                )
                data = bytearray(size * len(values))
                for index, (start, end) in enumerate(values):
                    struct.pack_into("=I", data, index * size, 10)
                    struct.pack_into("=Q", data, index * size + fields[1][1], start)
                    struct.pack_into("=Q", data, index * size + fields[2][1], end)
                data.extend(suffix)
                ptr, capacity, max_records = (
                    ctypes.c_void_p(),
                    ctypes.c_size_t(),
                    ctypes.c_size_t(),
                )
                request(
                    ctypes.byref(ptr),
                    ctypes.byref(capacity),
                    ctypes.byref(max_records),
                    None,
                )
                ctypes.memmove(ptr, bytes(data), len(data))
                complete(ptr, capacity.value, len(data), ctypes.byref(info))
                # Callback-owned metadata may change or disappear before decoding.
                layout.recordSize = 0
                entries[1].offset = 4096
            native.start_decoder()
            native.stop_decoder()
            groups, _ = native.drain_decoded()
            if invalid_size is not None:
                self.assertEqual(groups, [])
                self.assertEqual(native.decoder_max_sync_ns(), 0)
                return
            self.assertEqual(len(groups), 1)
            kind, columns = groups[0]
            self.assertEqual(kind, 10)
            self.assertEqual(struct.unpack("=3Q", columns[1][1]), (1, 3, 5))
            self.assertEqual(struct.unpack("=3Q", columns[2][1]), (2, 4, 6))
            self.assertEqual(native.decoder_max_sync_ns(), 6)
            self.assertEqual(native.pending_buffers(), 0)
        finally:
            native.stop_decoder()
            native.reset_buffers()


instantiate_parametrized_tests(TestCuspyDecoder)


class TestCuspyDecoderCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    @unittest.skipIf(
        not TEST_CUPTI_V13_5, "requires CUDA >= 13.5 + a loaded libcupti >= 13.5"
    )
    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_decoder_groups_distinct_layouts(self):
        # The native decode worker keys accumulated columns by (kind, layout) -- the
        # sorted (field_id, size) set of the captured record layout. When a kind's
        # field selection changes mid-session its records get a different layout, so
        # they must accumulate in a SEPARATE group (each internally length-aligned)
        # rather than corrupting one another's columns. Drive two selections for
        # CONCURRENT_KERNEL through the worker and assert drain returns two
        # length-consistent groups for the kind.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cuspy.cupti_python import CuptiError
        from torch.profiler._cuspy.records import Kernel

        native = torch._C._profiler._cuspy
        torch.cuda.init()
        lib = pylibcupti()
        kernel = ActivityKind.CONCURRENT_KERNEL

        native.reset_buffers()
        native.configure_buffers(4 << 20)
        try:
            sub = lib.subscribe()
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")

        def run_and_flush():
            x = torch.randn(256, 256, device="cuda")
            for _ in range(3):
                x = torch.relu(x @ x)
            x.sum().item()
            torch.cuda.synchronize()
            lib.activity_flush_all()
            for _ in range(250):  # wait (<= 5s) for the off-thread worker to drain
                if native.pending_buffers() == 0:
                    break
                time.sleep(0.02)
            time.sleep(0.1)  # let the last pulled buffer finish decoding

        try:
            lib.arm_user_defined_records(
                sub,
                native.buffer_request_callback_address(),
                native.buffer_complete_callback_address(),
            )
            native.configure_decoder(sub, lib.get_next_record_fn_address(), 0, -1)
            native.start_decoder()
            native.drain_decoded()  # clear residue

            lib.activity_enable(sub, kernel, [int(Kernel.START), int(Kernel.END)])
            run_and_flush()
            # change the selection -> a different layout for the same kind.
            lib.activity_disable(sub, kernel)
            lib.activity_enable(
                sub,
                kernel,
                [int(Kernel.START), int(Kernel.END), int(Kernel.CORRELATION_ID)],
            )
            run_and_flush()

            groups, _ = native.drain_decoded()
            kernel_groups = [cols for (k, cols) in groups if k == int(kernel)]
            # two distinct layouts for the kind -> two separately-keyed groups.
            self.assertEqual(len(kernel_groups), 2)
            self.assertEqual(
                sorted(tuple(sorted(g)) for g in kernel_groups),
                [
                    (0, int(Kernel.START), int(Kernel.END)),
                    (
                        0,
                        int(Kernel.START),
                        int(Kernel.END),
                        int(Kernel.CORRELATION_ID),
                    ),
                ],
            )
            # within each group every column has the same record count.
            for cols in kernel_groups:
                counts = {len(b) // sz for sz, b in cols.values() if sz}
                self.assertEqual(len(counts), 1)
                self.assertGreater(next(iter(counts)), 0)
        finally:
            native.stop_decoder()
            try:
                lib.disarm_user_defined_records(sub)
            except CuptiError:
                pass
            lib.unsubscribe(sub)


if __name__ == "__main__":
    run_tests()

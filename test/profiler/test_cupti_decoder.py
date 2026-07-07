# Owner(s): ["oncall: profiler"]

import time
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUPTI, TEST_CUPTI_V13_3
from torch.testing._internal.common_utils import run_tests, TestCase


# The decode tests drive the real libcupti (pylibcupti); guard its import on TEST_CUPTI
# (cupti-python imports its enums at load). TEST_CUPTI_V13_3 additionally requires a loaded
# libcupti >= 13.3 (the v2 user-defined-record API the decode worker drives).
if TEST_CUPTI:
    from torch.profiler._cupti.cupti_python import pylibcupti


class TestCuptiDecoder(TestCase):
    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires a loaded libcupti >= 13.3")
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

        from torch.profiler._cupti.cupti_python import CuptiError
        from torch.profiler._cupti.records import Kernel

        native = torch._C._profiler._cupti_monitor
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

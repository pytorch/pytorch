# Owner(s): ["oncall: profiler"]

import ctypes
import subprocess
import sys
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


# cupti-python imports its enums at module load, so the import itself gates on the
# package being installed.
try:
    from torch.profiler._cupti.cupti_python import CUPTI_SUCCESS, CuptiError, pylibcupti

    _HAS_CUPTI = True
except ModuleNotFoundError:
    _HAS_CUPTI = False


def _cupti_version() -> int:
    if not _HAS_CUPTI:
        return 0
    try:
        return pylibcupti().get_version()
    except Exception:
        return 0


# These tests drive the real libcupti through _PyLibCupti. The wrapper targets the
# v2 user-defined-record API, so gate on the same >= 13.3 floor the monitor uses:
# only run when an adequate libcupti is actually loaded (e.g. the 13.3 wheel is
# LD_PRELOADed); the torch-bundled libcupti is typically 13.1 and skips.
TEST_CUPTI_V13_3 = _cupti_version() >= 130300


@unittest.skipIf(not TEST_CUPTI_V13_3, "requires a loaded libcupti >= 13.3")
class TestPyLibCupti(TestCase):
    def test_get_version(self):
        self.assertGreaterEqual(pylibcupti().get_version(), 130300)

    def test_get_next_record_fn_address(self):
        # cuptiActivityGetNextRecord_v2 is present on >= 13.2, so its address (used
        # by the native decode worker) is nonzero here.
        addr = pylibcupti().get_next_record_fn_address()
        self.assertGreater(addr, 0)
        # ...and it is the live function, not garbage: invoke it through the address
        # the same way the native worker does -- int(subscriber, buffer, validSize,
        # &record) -> CUptiResult. With no subscriber/buffer CUPTI returns a defined
        # non-success status (CUPTI_ERROR_INVALID_PARAMETER) rather than crashing.
        get_next = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        )(addr)
        record = ctypes.c_void_p()
        self.assertNotEqual(
            get_next(None, None, 0, ctypes.byref(record)), CUPTI_SUCCESS
        )

    def test_result_string_and_check(self):
        lib = pylibcupti()
        # A successful rc never raises and decodes to CUPTI's real success string --
        # the actual cuptiGetResultString lookup, not the "rc=<n>" fallback.
        lib._check(CUPTI_SUCCESS, "noop")
        self.assertEqual(lib._result_string(CUPTI_SUCCESS), "CUPTI_SUCCESS")
        # A non-success rc raises CuptiError carrying the op name and the decoded
        # CUPTI error string (rc=1 -> CUPTI_ERROR_INVALID_PARAMETER).
        with self.assertRaises(CuptiError) as cm:
            lib._check(1, "deliberately_bad")
        self.assertIn("deliberately_bad", str(cm.exception))
        self.assertIn("CUPTI_ERROR_INVALID_PARAMETER", str(cm.exception))

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_v2_subscribe_timestamp_roundtrip(self):
        torch.cuda.init()
        lib = pylibcupti()
        try:
            sub_handle = lib.subscribe()
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        try:
            # get_timestamp's _v2 form requires an active subscriber and shares the
            # activity-record timebase, so it returns a positive monotonic value.
            self.assertGreater(lib.get_timestamp(sub_handle), 0)
            # best-effort per-subscriber toggle: returns a bool, never raises.
            self.assertIsInstance(
                lib.enable_kernel_latency_timestamps(sub_handle, True), bool
            )
        finally:
            lib.unsubscribe(sub_handle)

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_v2_activity_lifecycle(self):
        # Drive the subscription-scoped wrapper surface against real libcupti, the
        # same sequence the monitor runs: push/pop external correlation, arm UDR
        # with the native buffer callbacks, enable/disable kinds with field
        # selections, flush, and read the dropped-record count.
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.records import FIELD_REGISTRY

        monitor_ext = torch._C._profiler._cupti_monitor
        torch.cuda.init()
        lib = pylibcupti()
        try:
            sub = lib.subscribe()
        except CuptiError as e:
            self.skipTest(f"v2 subscribe unavailable on this driver/cupti: {e}")
        try:
            # external-correlation push/pop round-trips the id (v2 form, under sub).
            self.assertTrue(
                lib.activity_push_external_correlation_id(4242, sub_handle=sub)
            )
            self.assertEqual(
                lib.activity_pop_external_correlation_id(sub_handle=sub), 4242
            )
            lib.arm_user_defined_records(
                sub,
                monitor_ext.buffer_request_callback_address(),
                monitor_ext.buffer_complete_callback_address(),
            )
            # CONCURRENT_KERNEL + RUNTIME; enabling RUNTIME also exercises
            # disable_noisy_runtime_apis (best-effort, inside activity_enable).
            kinds = [ActivityKind.CONCURRENT_KERNEL, ActivityKind.RUNTIME]
            for kind in kinds:
                lib.activity_enable(sub, kind, FIELD_REGISTRY[kind])
            torch.randn(4096, device="cuda").sum()
            torch.cuda.synchronize()
            lib.activity_flush_all()  # must not raise
            self.assertGreaterEqual(lib.activity_get_num_dropped_records(0, 0), 0)
            for kind in kinds:
                lib.activity_disable(sub, kind)
            lib.disarm_user_defined_records(sub)
        finally:
            lib.unsubscribe(sub)

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_activity_enable_hw_trace_toggle(self):
        # HW-trace is a GLOBAL toggle -- it perturbs process-wide CUPTI state (and
        # breaks a sibling UDR session), so exercise it in an isolated child that
        # inherits this process's libcupti via LD_PRELOAD: enable then restore to
        # the default disabled state. exit(2) means the platform rejects HW trace.
        script = (
            "import sys, torch\n"
            "torch.cuda.init()\n"
            "from torch.profiler._cupti.cupti_python import pylibcupti, CuptiError\n"
            "lib = pylibcupti()\n"
            "try:\n"
            "    lib.activity_enable_hw_trace(True)\n"
            "except CuptiError:\n"
            "    sys.exit(2)\n"
            "lib.activity_enable_hw_trace(False)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        if proc.returncode == 2:
            self.skipTest("HW trace unsupported on this platform")
        self.assertEqual(proc.returncode, 0, proc.stderr)

    @unittest.skipIf(not torch.cuda.is_available(), "needs a CUDA context")
    def test_finalize_in_subprocess(self):
        # finalize() is cuptiFinalize -- a global, process-wide teardown the monitor
        # never calls. Run it in a child (which inherits this process's libcupti via
        # LD_PRELOAD) so it can't tear CUPTI down for sibling tests; a clean exit
        # means the wrapper's call + rc-check succeeded.
        script = (
            "import torch; torch.cuda.init(); "
            "from torch.profiler._cupti.cupti_python import pylibcupti; "
            "pylibcupti().finalize()"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    run_tests()

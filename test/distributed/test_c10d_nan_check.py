# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the NaN check hook
# (torch._C._distributed_c10d.NanCheckHook): NaN checking of collective input
# buffers driven by the ProcessGroup pre-hooks rather than native backend
# support, so it works for any backend routed through the c10d ops.

import os
import signal
import sys

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from c10d_backend_common import (
    C10D_BACKENDS,
    C10dBackendTest,
    instantiate_backend_tests,
)

from torch._C._distributed_c10d import NanCheckHook
from torch.testing._internal.common_utils import run_tests


# Tests in which the check is expected to fire.
DETECTING_TESTS = ("test_nan_in_input_detected", "test_env_var_auto_attach")


class AbstractNanCheckHookTest(C10dBackendTest):
    def setUp(self):
        super().setUp()
        # Set after super().setUp(): MultiProcessTestCase.setUp() resets the
        # dict, and the parent only consults it when joining the children.
        if self.device_type == "cuda" and self._testMethodName in DETECTING_TESTS:
            # CUDA and ROCm can report the device trap as a runtime exception.
            # _assert_nan_detected normalizes that path with os._exit(SIGABRT).
            self.special_return_code_checks = {
                getattr(self, self._testMethodName).__wrapped__: signal.SIGABRT
            }

    def _assert_nan_detected(self, tensor):
        if self.device_type == "cuda":
            try:
                dist.all_reduce(tensor)
                torch.cuda.synchronize()
            except Exception:
                # os._exit, not sys.exit: the CUDA context is poisoned, so
                # unwinding through tearDown's destroy_process_group can hang
                # waiting on collectives that will never complete.
                os._exit(signal.SIGABRT)
            self.fail("NaN in collective input was not detected")
        else:
            with self.assertRaisesRegex(RuntimeError, "NaN"):
                dist.all_reduce(tensor)

    def test_nan_in_input_detected(self):
        self._init_pg()
        # The handle is deliberately dropped: the group owns the hook.
        NanCheckHook.attach(dist.group.WORLD)
        # All ranks poison their input so no rank is left waiting on a peer
        # that died in the check.
        t = torch.ones(1024, device=self.device)
        t[42] = float("nan")
        self._assert_nan_detected(t)

    def test_nan_in_recv_buffer_ok(self):
        self._init_pg()
        hook = NanCheckHook.attach(dist.group.WORLD)

        # Non-root broadcast buffers are receive buffers.
        t = torch.ones(3, 4, device=self.device)
        if self.rank != 0:
            t[1, 1] = float("nan")
        dist.broadcast(t, 0)

        send = torch.ones(4, device=self.device)
        recv = torch.full((4,), float("nan"), device=self.device)
        peer = 1 - self.rank
        if self.rank == 0:
            dist.send(send, peer)
            dist.recv(recv, peer)
        else:
            dist.recv(recv, peer)
            dist.send(send, peer)

        out = [
            torch.full((4,), float("nan"), device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(out, send)

        expected = torch.ones(4, device=self.device)
        self.assertEqual(recv, expected)
        self.assertEqual(t, torch.ones(3, 4, device=self.device))
        for o in out:
            self.assertEqual(o, expected)
        hook.remove()

    def test_legitimate_data_not_flagged(self):
        self._init_pg()
        hook = NanCheckHook.attach(dist.group.WORLD)

        t = torch.full((8,), float(self.rank + 1), device=self.device)
        dist.all_reduce(t)
        self.assertEqual(t, torch.full((8,), 3.0, device=self.device))

        # Non-floating dtypes are skipped by the checker.
        i = torch.ones(4, dtype=torch.int64, device=self.device)
        dist.all_reduce(i)
        self.assertEqual(i, torch.full((4,), 2, dtype=torch.int64, device=self.device))

        dist.barrier()
        hook.remove()

    def test_off_by_default_and_removable(self):
        self._init_pg()

        t = torch.ones(8, device=self.device)
        t[0] = float("nan")
        dist.all_reduce(t)

        NanCheckHook.attach(dist.group.WORLD).remove()
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

    def test_env_var_auto_attach(self):
        os.environ["TORCH_DIST_NAN_CHECK"] = "1"
        try:
            self._init_pg()
        finally:
            del os.environ["TORCH_DIST_NAN_CHECK"]

        # Nothing on the Python side holds the auto-attached handle, so this
        # also covers the hook outliving it.
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        t[0] = float("nan")
        self._assert_nan_detected(t)


instantiate_backend_tests(
    globals(), "NanCheckHook", AbstractNanCheckHookTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()

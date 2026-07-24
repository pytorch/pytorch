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
from torch.distributed.distributed_c10d import _world
from torch.testing._internal.common_utils import run_tests


class AbstractNanCheckHookTest(C10dBackendTest):
    def setUp(self):
        super().setUp()
        # Set after super().setUp(): MultiProcessTestCase.setUp() resets the
        # dict, and the parent only consults it when joining the children.
        if (
            self.device_type == "cuda"
            and self._testMethodName == "test_nan_in_input_detected"
        ):
            # The CUDA checker traps on the device rather than raising a
            # catchable error, so the process dies instead; see
            # test_c10d_nccl.py's test_nan_assert. ROCm's assert(0) surfaces as
            # a signal rather than a clean exit code.
            self.special_return_code_checks = {
                self.test_nan_in_input_detected.__wrapped__: (
                    -signal.SIGABRT if torch.version.hip else signal.SIGABRT
                )
            }

    def test_nan_in_input_detected(self):
        self._init_pg()
        hook = NanCheckHook.attach(dist.group.WORLD)
        # All ranks poison their input so no rank is left waiting on a peer
        # that died in the check.
        t = torch.ones(1024, device=self.device)
        t[42] = float("nan")

        if self.device_type == "cuda":
            try:
                dist.all_reduce(t)
                torch.cuda.synchronize()
            except Exception:
                sys.exit(signal.SIGABRT)
            self.fail("NaN in collective input was not detected")
        else:
            with self.assertRaisesRegex(RuntimeError, "NaN"):
                dist.all_reduce(t)
            hook.remove()

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
        self.assertEqual(len(_world.nan_check_hooks), 0)

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
        self.assertIn(dist.group.WORLD, _world.nan_check_hooks)

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        dist.destroy_process_group()
        self.assertEqual(len(_world.nan_check_hooks), 0)


instantiate_backend_tests(
    globals(), "NanCheckHook", AbstractNanCheckHookTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()

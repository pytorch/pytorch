# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the NaN check hook
# (torch._C._distributed_c10d.NanCheckHook): NaN checking of collective input
# buffers driven by the ProcessGroup pre-hooks rather than native backend
# support, so it works for any backend routed through the c10d ops.

import gc
import os
import signal
import sys
from datetime import timedelta

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

from torch._C._distributed_c10d import NanCheckHook, ReconfigureOptions
from torch.testing._internal.common_distributed import core_dumps_disabled
from torch.testing._internal.common_utils import run_tests


# Tests in which the check is expected to fire.
DETECTING_TESTS = (
    "test_nan_in_input_detected",
    "test_env_var_auto_attach",
    "test_split_group_env_var_auto_attach",
    "test_split_group_nccl_env_var",
    "test_nccl_env_var",
    "test_set_enable_nan_check",
    "test_set_enable_nan_check_toggle",
    "test_set_enable_nan_check_new_group",
    "test_nccl_default_backend_env_var",
    "test_nccl_env_var_cuda_only",
    "test_set_enable_nan_check_debug_wrapper",
    "test_set_enable_nan_check_shrink_group",
    "test_nccl_env_var_coalesced",
)


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

    def _skip_unless_set_enable_nan_check(self):
        # ProcessGroupNCCL toggles its native check; ProcessGroupNCCL2 toggles
        # the hook. nccl-lazy wraps ProcessGroupNCCL2 without exposing the setter.
        if self.backend_name not in ("nccl-legacy", "nccl2"):
            self.skipTest(f"{self.backend_name} has no _set_enable_nan_check")

    def _skip_unless_nan_check_hook_backend(self):
        if self.backend_name not in ("nccl2", "nccl-lazy"):
            self.skipTest(f"{self.backend_name} does not use the hook for NCCL")

    def _init_pg_as(self, backend):
        # _init_pg with a different backend string.
        os.environ["LOCAL_RANK"] = str(self.rank)
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=dist.FileStore(self.file_name, self.world_size),
            timeout=timedelta(seconds=60),
        )

    def _assert_nan_detected(self, tensor, group=None):
        if self.device_type == "cuda":
            with core_dumps_disabled():
                try:
                    dist.all_reduce(tensor, group=group)
                    torch.cuda.synchronize()
                except Exception:
                    # os._exit, not sys.exit: the CUDA context is poisoned, so
                    # unwinding through tearDown's destroy_process_group can hang
                    # waiting on collectives that will never complete.
                    os._exit(signal.SIGABRT)
            self.fail("NaN in collective input was not detected")
        else:
            with self.assertRaisesRegex(RuntimeError, "NaN"):
                dist.all_reduce(tensor, group=group)

    def test_nan_in_input_detected(self):
        self._init_pg()
        # The handle is deliberately dropped: the group owns the hook.
        NanCheckHook.attach(dist.group.WORLD)
        # Initialize lazy GPU backends before the device assert poisons the stream.
        t = torch.ones(1024, device=self.device)
        dist.all_reduce(t)
        # All ranks poison their input so no rank is left waiting on a peer
        # that died in the check.
        t[42] = float("nan")
        self._assert_nan_detected(t)

    def test_nan_in_recv_buffer_ok(self):
        self._init_pg()
        hook = NanCheckHook.attach(dist.group.WORLD)
        self._check_recv_buffers_not_flagged()
        hook.remove()

    def test_nccl_env_var_recv_buffer_ok(self):
        if not self.backend_name.startswith("nccl"):
            self.skipTest("TORCH_NCCL_NAN_CHECK is consumed by the NCCL backends")
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            self._init_pg()
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]
        self._check_recv_buffers_not_flagged()
        torch.cuda.synchronize()

    def _check_recv_buffers_not_flagged(self):
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

    def test_split_group_env_var_auto_attach(self):
        self._init_pg()
        default_pg = dist.distributed_c10d._get_default_group()
        if not default_pg._get_backend(self.device).supports_splitting:
            self.skipTest(f"{self.backend_name} does not support split_group")
        if self.device_type != "cpu":
            default_pg.bound_device_id = self.device
        # Initialize the parent communicator before splitting.
        dist.all_reduce(torch.ones(8, device=self.device))

        # Only the child is created with the check enabled.
        os.environ["TORCH_DIST_NAN_CHECK"] = "1"
        try:
            child = dist.split_group(split_ranks=[list(range(self.world_size))])
        finally:
            del os.environ["TORCH_DIST_NAN_CHECK"]

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t, group=child)
        t[0] = float("nan")
        self._assert_nan_detected(t, group=child)

    def test_split_group_nccl_env_var(self):
        if not self.backend_name.startswith("nccl"):
            self.skipTest("TORCH_NCCL_NAN_CHECK is consumed by the NCCL backends")
        self._init_pg()
        default_pg = dist.distributed_c10d._get_default_group()
        if not default_pg._get_backend(self.device).supports_splitting:
            self.skipTest(f"{self.backend_name} does not support split_group")
        default_pg.bound_device_id = self.device
        dist.all_reduce(torch.ones(8, device=self.device))

        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            child = dist.split_group(split_ranks=[list(range(self.world_size))])
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t, group=child)
        t[0] = float("nan")
        self._assert_nan_detected(t, group=child)

    def test_nccl_env_var(self):
        if not self.backend_name.startswith("nccl"):
            self.skipTest("TORCH_NCCL_NAN_CHECK is consumed by the NCCL backends")
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            self._init_pg()
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        t[0] = float("nan")
        self._assert_nan_detected(t)

    def test_set_enable_nan_check(self):
        self._skip_unless_set_enable_nan_check()
        self._init_pg()
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)

        dist.get_backend_impl(device=self.device)._set_enable_nan_check(True)
        t[0] = float("nan")
        self._assert_nan_detected(t)

    def test_set_enable_nan_check_toggle(self):
        self._skip_unless_set_enable_nan_check()
        self._init_pg()
        backend = dist.get_backend_impl(device=self.device)
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)

        backend._set_enable_nan_check(True)
        backend._set_enable_nan_check(False)
        t[0] = float("nan")
        dist.all_reduce(t)
        torch.cuda.synchronize()

        backend._set_enable_nan_check(True)
        self._assert_nan_detected(t)

    def test_set_enable_nan_check_new_group(self):
        self._skip_unless_set_enable_nan_check()
        self._init_pg()
        group = dist.new_group(list(range(self.world_size)))
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t, group=group)

        group._get_backend(self.device)._set_enable_nan_check(True)
        t[0] = float("nan")
        # Only the subgroup is checked.
        dist.all_reduce(t.clone())
        torch.cuda.synchronize()
        self._assert_nan_detected(t, group=group)

    def test_nccl_default_backend_env_var(self):
        # "nccl" resolves to nccl2 unless TORCH_DIST_USE_NCCL2=0.
        if self.backend_name != "nccl2":
            self.skipTest("covers backend='nccl' once")
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        os.environ["TORCH_DIST_USE_NCCL2"] = "1"
        try:
            self._init_pg_as("nccl")
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]
            del os.environ["TORCH_DIST_USE_NCCL2"]
        backend = dist.group.WORLD._get_backend(self.device)
        self.assertEqual(backend.name(), "nccl2")

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        t[0] = float("nan")
        self._assert_nan_detected(t)

    def test_nccl_env_var_cuda_only(self):
        self._skip_unless_nan_check_hook_backend()
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            self._init_pg_as(f"cpu:gloo,cuda:{self.backend_name}")
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]

        # Like the native check, only the NCCL backend's operations are checked.
        t = torch.ones(8)
        t[0] = float("nan")
        dist.all_reduce(t)
        self.assertTrue(t[0].isnan())
        self._assert_nan_detected(t.to(self.device))

    def test_nccl_env_var_other_current_device(self):
        # The check runs on the tensor's device, not the current one.
        self._skip_unless_nan_check_hook_backend()
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            self._init_pg()
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]
        t = torch.ones(8, device=self.device)
        torch.cuda.set_device(1 - self.rank)
        dist.all_reduce(t)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.full_like(t, self.world_size))

    def test_nccl_env_var_non_dense_input(self):
        # Layouts the kernel can't scan are left to the backend's validation.
        self._skip_unless_nan_check_hook_backend()
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            self._init_pg()
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]
        t = torch.ones(1, device=self.device).expand(1 << 20)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            dist.all_reduce(t)

    def test_nccl_env_var_coalesced(self):
        # Unlike ProcessGroupNCCL's native check, coalesced ops are checked.
        self._skip_unless_nan_check_hook_backend()
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            self._init_pg()
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]
        a = torch.ones(8, device=self.device)
        b = torch.ones(8, device=self.device)
        # Clean run first, so only the NaN can make the second one fail.
        with dist._coalescing_manager(device=self.device):
            dist.all_reduce(a)
            dist.all_reduce(b)
        torch.cuda.synchronize()
        b[0] = float("nan")
        with core_dumps_disabled():
            try:
                with dist._coalescing_manager(device=self.device):
                    dist.all_reduce(a)
                    dist.all_reduce(b)
                torch.cuda.synchronize()
            except Exception:
                os._exit(signal.SIGABRT)
        self.fail("NaN in coalesced input was not detected")

    def test_nccl_env_var_invalid(self):
        self._skip_unless_nan_check_hook_backend()
        os.environ["TORCH_NCCL_NAN_CHECK"] = "on"
        try:
            with self.assertRaisesRegex(ValueError, "TORCH_NCCL_NAN_CHECK"):
                self._init_pg()
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]

    def test_set_enable_nan_check_disables_env_var(self):
        self._skip_unless_set_enable_nan_check()
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        try:
            self._init_pg()
        finally:
            del os.environ["TORCH_NCCL_NAN_CHECK"]

        dist.get_backend_impl(device=self.device)._set_enable_nan_check(False)
        t = torch.ones(8, device=self.device)
        t[0] = float("nan")
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertTrue(t[0].isnan())

    def test_set_enable_nan_check_after_destroy(self):
        # The group owns the check: destroying the group frees it, and with it
        # the check. Hooks used to keep the group alive.
        if self.backend_name != "nccl2":
            self.skipTest("only nccl2 toggles NanCheckHook")
        self._init_pg()
        group = dist.new_group(list(range(self.world_size)))
        backend = group._get_backend(self.device)
        backend._set_enable_nan_check(True)

        dist.destroy_process_group(group)
        del group
        gc.collect()
        with self.assertRaisesRegex(RuntimeError, "No NaN check is attached"):
            backend._set_enable_nan_check(True)

    def test_set_enable_nan_check_shrink_group(self):
        if self.backend_name != "nccl2":
            self.skipTest("only nccl2 toggles NanCheckHook")
        self._init_pg()
        if not dist.get_backend_impl(device=self.device).supports_shrinking:
            self.skipTest("nccl2 shrink requires NCCL 2.27 or later")
        group = dist.new_group(device_id=self.device)
        dist.barrier(group=group)
        excluded = list(range(1, self.world_size))
        if self.rank in excluded:
            dist.destroy_process_group(group)
            # Skip teardown: it would wait on rank 0, which dies in the check.
            os._exit(0)

        shrunk = dist.shrink_group(excluded, group=group)
        shrunk._get_backend(self.device)._set_enable_nan_check(True)
        t = torch.ones(8, device=self.device)
        t[0] = float("nan")
        self._assert_nan_detected(t, group=shrunk)

    def test_set_enable_nan_check_reconfigure(self):
        # Reconfigure swaps the ranks; broadcast must check the new root only.
        if self.backend_name != "nccl2":
            self.skipTest("only nccl2 toggles NanCheckHook")
        # Same requirements as ProcessGroupNCCL2's enable_reconfigure check.
        nccl_version = torch.cuda.nccl.version()
        if nccl_version < (2, 28) or (torch.version.hip and nccl_version < (2, 30, 7)):
            self.skipTest("nccl2 reconfigure requires NCCL 2.28+ / RCCL 2.30.7+")
        self._init_pg()
        opts = dist.ProcessGroupNCCL2.Options()
        opts.enable_reconfigure = True
        group = dist.new_group(backend="nccl2", pg_options=opts, device_id=self.device)
        backend = group._get_backend(self.device)
        handles = dist.PrefixStore(
            "nan_reconf", dist.distributed_c10d._get_default_store()
        )
        handles.set(str(self.rank), backend.get_reconfigure_handle())
        reconfigure_opts = ReconfigureOptions()
        reconfigure_opts.uuid = 1
        reconfigure_opts.handles = [
            handles.get(str(rank)).decode() for rank in reversed(range(self.world_size))
        ]
        reconfigure_opts.timeout = timedelta(seconds=30)
        group.reconfigure(reconfigure_opts).wait()
        self.assertEqual(group.rank(), self.world_size - 1 - self.rank)

        backend._set_enable_nan_check(True)
        t = torch.ones(8, device=self.device)
        if group.rank() != 0:
            # Receive buffer: must not be checked.
            t[0] = float("nan")
        dist.broadcast(t, group_src=0, group=group)
        torch.cuda.synchronize(self.device)
        self.assertEqual(t, torch.ones_like(t))
        dist.destroy_process_group(group)

    def test_set_enable_nan_check_debug_wrapper(self):
        # TORCH_DISTRIBUTED_DEBUG=DETAIL wraps the backend; the check belongs to
        # the wrapped one.
        if self.backend_name != "nccl2":
            self.skipTest("only nccl2 toggles NanCheckHook")
        prev = dist.get_debug_level()
        dist.set_debug_level(dist.DebugLevel.DETAIL)
        try:
            self._init_pg()
        finally:
            dist.set_debug_level(prev)
        backend = dist.get_backend_impl(device=self.device)
        self.assertIsInstance(backend, dist.distributed_c10d._ProcessGroupWrapper)
        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)

        backend.wrapped_pg._set_enable_nan_check(True)
        t[0] = float("nan")
        self._assert_nan_detected(t)


instantiate_backend_tests(
    globals(), "NanCheckHook", AbstractNanCheckHookTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()

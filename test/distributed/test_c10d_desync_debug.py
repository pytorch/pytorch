# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the desync debug hook
# (torch._C._distributed_c10d.DesyncDebugHook): ProcessGroupNCCL's
# TORCH_NCCL_DESYNC_DEBUG report, built on the ProcessGroup hooks.

import os
import sys
import threading
import time
import warnings
from datetime import timedelta
from unittest import mock

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

from torch._C._distributed_c10d import DesyncDebugHook
from torch.distributed.distributed_c10d import _get_process_group_store
from torch.testing._internal.common_utils import run_tests


TIMEOUT_S = 60


def _start_key(rank):
    return f"c10d_desync_{rank}_trace_start"


def _end_key(rank):
    return f"c10d_desync_{rank}_trace_end"


def _published_seq(store, key):
    if not store.check([key]):
        return 0
    return int.from_bytes(store.get(key)[:8], sys.byteorder)


def _wait_published(store, key, seq):
    # The writer flushes asynchronously, so the only signal is the store itself.
    deadline = time.monotonic() + TIMEOUT_S
    while _published_seq(store, key) < seq:
        if time.monotonic() > deadline:
            raise TimeoutError(f"{key} never reached seq {seq}")


class _GatedStore(dist.Store):
    """Forwards to `store`; set() blocks while the gate is closed."""

    def __init__(self, store):
        super().__init__()
        self._store = store
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.sets = 0
        self.fail = False

    def set(self, key, value):
        self.sets += 1
        self.entered.set()
        self.gate.wait()
        if self.fail:
            raise RuntimeError("store down")
        self._store.set(key, value)

    def get(self, key):
        return self._store.get(key)

    def check(self, keys):
        return self._store.check(keys)

    def add(self, key, value):
        return self._store.add(key, value)

    def wait(self, keys, timeout=None):
        if timeout is None:
            return self._store.wait(keys)
        return self._store.wait(keys, timeout)


class AbstractDesyncDebugHookTest(C10dBackendTest):
    def _base_store(self):
        return dist.FileStore(self.file_name, self.world_size)

    def test_report_names_lagging_rank(self):
        self._init_pg()
        pg = dist.group.WORLD
        # Attached on every rank before the group's first op, so sequence
        # numbers line up.
        hook = DesyncDebugHook.attach(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        # Only rank 0 sends and only rank 1 receives: p2p must not shift the
        # collective numbering.
        if self.rank == 0:
            dist.send(t, 1)
        else:
            dist.recv(t, 0)
        dist.all_reduce(t)

        store = _get_process_group_store(pg)
        base = self._base_store()
        if self.rank == 1:
            # Rank 1 stops after collective #2; rank 0 goes on to #3.
            _wait_published(store, _start_key(1), 2)
            if pg.supports_completion_hooks:
                _wait_published(store, _end_key(1), 2)
            base.set("rank1_published", b"1")
            base.wait(["reported"])
            dist.all_reduce(t)
            return

        base.wait(["rank1_published"])
        work = dist.all_reduce(t, async_op=True)
        try:
            report = hook.report()
        finally:
            base.set("reported", b"1")
        work.wait()

        self.assertIn("[0] Timeout at collective: all_reduce, #3", report)
        # Backends with completion hooks know rank 1 finished #2; the others
        # only know it started it.
        if pg.supports_completion_hooks:
            self.assertIn(
                "[1] finished collective #2, but didn't join collective #3", report
            )
        else:
            self.assertIn("[1] joined but didn't finish collective #2", report)
        hook.remove()

    def test_report_without_pending_op_raises(self):
        self._init_pg()
        hook = DesyncDebugHook.attach(dist.group.WORLD)
        # No op issued: there is nothing to report on.
        with self.assertRaisesRegex(RuntimeError, "no collective pending"):
            hook.report(timedelta(seconds=TIMEOUT_S))
        hook.remove()
        with self.assertRaisesRegex(RuntimeError, "removed"):
            hook.report()

    def _gated_pg(self):
        self._init_pg()
        # A group sharing WORLD's backend but publishing through a store whose
        # writes can be held or failed, standing in for a slow or dead TCPStore.
        store = _GatedStore(dist.PrefixStore("gated/", self._base_store()))
        pg = dist.ProcessGroup(store, self.rank, self.world_size)
        backend = dist.group.WORLD._get_backend(self.device)
        pg._register_backend(self.device, dist.ProcessGroup.BackendType.CUSTOM, backend)
        pg._set_default_backend(dist.ProcessGroup.BackendType.CUSTOM)
        return pg, store

    def test_store_never_blocks_issue(self):
        pg, store = self._gated_pg()
        store.gate.clear()
        hook = DesyncDebugHook.attach(pg)
        num_ops = 50
        t = torch.ones(4, device=self.device)
        try:
            pg.allreduce([t]).wait()
            # The writer is now stuck in its first set.
            self.assertTrue(store.entered.wait(TIMEOUT_S))
            for _ in range(num_ops - 1):
                pg.allreduce([t]).wait()
            self.assertEqual(store.sets, 1)
            store.gate.set()

            _wait_published(store, _start_key(self.rank), num_ops)
            # Coalesced: the latest start and end per flush, not two keys per op.
            self.assertLess(store.sets, num_ops // 5)
        finally:
            store.gate.set()
            hook.remove()

    def test_store_failure_reported(self):
        pg, store = self._gated_pg()
        store.fail = True
        hook = DesyncDebugHook.attach(pg)
        t = torch.ones(4, device=self.device)
        work = pg.allreduce([t])
        try:
            with self.assertRaisesRegex(RuntimeError, "store unavailable"):
                hook.report(timedelta(seconds=TIMEOUT_S))
        finally:
            work.wait()
            hook.remove()

    def test_no_tracking_during_cuda_graph_capture(self):
        # Collectives captured into a graph run at replay, not here, so they
        # are not numbered.
        if self.device_type != "cuda":
            self.skipTest("capture test is CUDA only")
        self._init_pg()
        pg = dist.group.WORLD
        t = torch.ones(4, device=self.device)
        # Warm up so comm initialization does not happen inside the capture.
        dist.all_reduce(t)
        torch.cuda.synchronize()
        hook = DesyncDebugHook.attach(pg)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dist.all_reduce(t)
        graph.replay()
        dist.all_reduce(t)
        torch.cuda.synchronize()

        # report() publishes the latest state first, whether or not it then
        # finds a pending collective.
        try:
            hook.report(timedelta(seconds=TIMEOUT_S))
        except RuntimeError:
            pass
        self.assertEqual(
            _published_seq(_get_process_group_store(pg), _start_key(self.rank)), 1
        )
        hook.remove()

    @mock.patch.dict(os.environ, {"TORCH_DIST_DESYNC_DEBUG": "1"})
    def test_env_var_auto_attach(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._init_pg()
            sub = dist.new_group(list(range(self.world_size)))

        pg = dist.group.WORLD
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        dist.all_reduce(t)
        dist.all_reduce(t, group=sub)
        dist.barrier()
        ignored = [w for w in caught if "TORCH_DIST_DESYNC_DEBUG" in str(w.message)]
        self.assertEqual(
            len(ignored) > 0,
            not (pg.supports_abort_hooks and sub.supports_abort_hooks),
        )
        for group, seq in ((pg, 2), (sub, 1)):
            store = _get_process_group_store(group)
            if group.supports_abort_hooks:
                # Nothing on the Python side holds the handle; the group does.
                _wait_published(store, _start_key(self.rank), seq)
            else:
                # Without an abort hook nothing would ever log the report.
                self.assertFalse(store.check([_start_key(self.rank)]))

    @mock.patch.dict(os.environ, {"TORCH_DIST_DESYNC_DEBUG": "0"})
    def test_off_by_default(self):
        self._init_pg()
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        dist.barrier()
        store = _get_process_group_store(dist.group.WORLD)
        self.assertFalse(store.check([_start_key(self.rank)]))


instantiate_backend_tests(
    globals(), "DesyncDebugHook", AbstractDesyncDebugHookTest, C10D_BACKENDS
)


if __name__ == "__main__":
    run_tests()

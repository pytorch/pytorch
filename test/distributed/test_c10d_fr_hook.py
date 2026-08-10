# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for the FlightRecorder hook
# (torch._C._distributed_c10d.FlightRecorderHook): FlightRecorder recording
# driven by the ProcessGroup pre/post collective hooks rather than native
# backend integration, so it works for any backend routed through c10d ops.
# Modeled on torchcomms' hooks/fr FlightRecorderTest; parameterized over
# backends like test_c10d_fault_tolerance.py.

import json
import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch._C._distributed_c10d import FlightRecorderHook
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


FR_HOOK_BACKENDS = [
    ("gloo", "cpu"),
    ("nccl2", "cuda"),
]


class AbstractFlightRecorderHookTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        if self.device_type == "cuda":
            return torch.device(f"cuda:{self.rank}")
        return torch.device(self.device_type)

    def setUp(self):
        super().setUp()
        # Note: gloo also records natively into the same recorder; keying
        # entries by the hook's profiling_name prefix ("c10d:") keeps the
        # assertions backend-agnostic.
        os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )
        return dist.group.WORLD

    def _hook_entries(self):
        trace = json.loads(torch._C._distributed_c10d._dump_fr_trace_json())
        return [
            e
            for e in trace.get("entries", [])
            if e["profiling_name"].startswith("c10d:")
        ]

    def test_records_and_retires_collectives(self):
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        dist.barrier()
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        names = [e["profiling_name"] for e in entries]
        self.assertIn("c10d:allreduce", names)
        self.assertIn("c10d:broadcast", names)
        self.assertIn("c10d:barrier", names)
        # Post-hooks fire after issue, so every recorded op must be retired.
        # With null start/end events (no GPU timing) the state stays
        # "scheduled"; retired is the completion signal, as with Gloo's
        # native FR recording.
        for e in entries:
            self.assertTrue(e["retired"], msg=str(e))
        hook.remove()

    def test_records_tensor_metadata(self):
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, 8, device=self.device)
        dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        allreduce = [e for e in entries if e["profiling_name"] == "c10d:allreduce"]
        self.assertEqual(len(allreduce), 1)
        self.assertEqual(allreduce[0]["input_sizes"], [[4, 8]])
        self.assertEqual(allreduce[0]["input_dtypes"], ["Float"])
        hook.remove()

    def test_p2p_and_collective_seq_ids(self):
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        send_t = torch.ones(4, device=self.device)
        recv_t = torch.empty(4, device=self.device)
        peer = 1 - self.rank
        if self.rank == 0:
            dist.send(send_t, peer)
            dist.recv(recv_t, peer)
        else:
            dist.recv(recv_t, peer)
            dist.send(send_t, peer)
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._hook_entries()[before:]
        p2p = [e for e in entries if e["profiling_name"] in ("c10d:send", "c10d:recv")]
        coll = [e for e in entries if e["profiling_name"] == "c10d:allreduce"]
        self.assertEqual(len(p2p), 2)
        self.assertEqual(len(coll), 1)
        # P2P ops advance p2p_seq_id only; collectives advance
        # collective_seq_id only.
        self.assertEqual(sorted(e["p2p_seq_id"] for e in p2p), [1, 2])
        self.assertEqual(coll[0]["collective_seq_id"], 1)
        hook.remove()

    def test_remove_stops_recording(self):
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        count_attached = len(self._hook_entries())
        self.assertGreater(count_attached, 0)

        hook.remove()
        dist.all_reduce(t)
        self.assertEqual(len(self._hook_entries()), count_attached)

    def test_multiple_collectives_entry_order(self):
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        for _ in range(5):
            dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        seqs = [
            e["collective_seq_id"]
            for e in entries
            if e["profiling_name"] == "c10d:allreduce"
        ]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), 5)
        hook.remove()


def _make_fr_hook_test_class(backend_name, device_type):
    class FlightRecorderHookTest(AbstractFlightRecorderHookTest, MultiProcessTestCase):
        pass

    FlightRecorderHookTest.backend_name = backend_name
    FlightRecorderHookTest.device_type = device_type
    FlightRecorderHookTest.__name__ = (
        f"{backend_name.capitalize()}FlightRecorderHookTest"
    )
    FlightRecorderHookTest.__qualname__ = FlightRecorderHookTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(FlightRecorderHookTest)
    if device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 2,
            "FR hook CUDA tests require at least 2 GPUs",
        )(cls)
    return cls


for backend_name, device_type in FR_HOOK_BACKENDS:
    globals()[f"{backend_name.capitalize()}FlightRecorderHookTest"] = (
        _make_fr_hook_test_class(backend_name, device_type)
    )


if __name__ == "__main__":
    run_tests()

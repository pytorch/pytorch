# Owner(s): ["oncall: distributed"]
#
# Backend-agnostic tests for automatic FlightRecorder hook recording.

import gc
import json
import os
import pickle
import sys
import time
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
    ("nccl", "cuda"),
    ("nccl2", "cuda"),
    ("nccl-lazy", "cuda"),
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
            if e["profiling_name"].startswith(f"{self.backend_name}:")
        ]

    def _wait_for_retired_entries(self, before, count):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            entries = self._hook_entries()[before:]
            if len(entries) >= count and all(e["retired"] for e in entries):
                return entries
            time.sleep(0.05)
        return self._hook_entries()[before:]

    def test_records_and_retires_collectives(self):
        self._init_pg()
        before = len(self._hook_entries())

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        dist.barrier()
        if self.device_type == "cuda":
            torch.cuda.synchronize()

        entries = self._wait_for_retired_entries(before, 3)
        names = [e["profiling_name"] for e in entries]
        self.assertIn(f"{self.backend_name}:all_reduce", names)
        self.assertIn(f"{self.backend_name}:broadcast", names)
        self.assertIn(f"{self.backend_name}:barrier", names)
        for e in entries:
            self.assertTrue(e["retired"], msg=str(e))
            self.assertEqual(e["state"], "completed")

    def test_disabled_does_not_record(self):
        os.environ["TORCH_FR_BUFFER_SIZE"] = "0"
        self._init_pg()
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        self.assertEqual(self._hook_entries(), [])

    def test_records_tensor_metadata(self):
        self._init_pg()
        before = len(self._hook_entries())

        t = torch.ones(4, 8, device=self.device)
        dist.all_reduce(t)

        entries = self._wait_for_retired_entries(before, 1)
        allreduce = [
            e
            for e in entries
            if e["profiling_name"] == f"{self.backend_name}:all_reduce"
        ]
        self.assertEqual(len(allreduce), 1)
        self.assertEqual(allreduce[0]["input_sizes"], [[4, 8]])
        self.assertEqual(allreduce[0]["input_dtypes"], ["Float"])
        self.assertEqual(allreduce[0]["output_sizes"], [[4, 8]])
        self.assertEqual(allreduce[0]["output_dtypes"], ["Float"])

    def test_p2p_and_collective_seq_ids(self):
        self._init_pg()
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

        entries = self._wait_for_retired_entries(before, 3)
        p2p = [
            e
            for e in entries
            if e["profiling_name"]
            in (f"{self.backend_name}:send", f"{self.backend_name}:recv")
        ]
        coll = [
            e
            for e in entries
            if e["profiling_name"] == f"{self.backend_name}:all_reduce"
        ]
        self.assertEqual(len(p2p), 2)
        self.assertEqual(len(coll), 1)
        # P2P ops advance p2p_seq_id only; collectives advance
        # collective_seq_id only.
        self.assertEqual(sorted(e["p2p_seq_id"] for e in p2p), [1, 2])
        self.assertEqual(coll[0]["collective_seq_id"], 1)
        for entry in entries:
            self.assertTrue(entry["retired"], msg=str(entry))
            self.assertEqual(entry["state"], "completed")

    def test_remove_stops_recording(self):
        pg = self._init_pg()
        # attach() returns the automatically installed hook.
        hook = FlightRecorderHook.attach(pg)
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        count_attached = len(self._hook_entries())
        self.assertGreater(count_attached, 0)

        hook.remove()
        dist.all_reduce(t)
        self.assertEqual(len(self._hook_entries()), count_attached)

    def test_hook_handle_outlives_process_group(self):
        if self.backend_name != "gloo":
            self.skipTest("hook lifetime is backend-independent")
        pg = self._init_pg()
        hook = FlightRecorderHook.attach(pg)
        dist.destroy_process_group()
        del pg
        gc.collect()
        hook.remove()

    def test_multiple_collectives_entry_order(self):
        self._init_pg()
        before = len(self._hook_entries())

        t = torch.ones(4, device=self.device)
        for _ in range(5):
            dist.all_reduce(t)

        entries = self._hook_entries()[before:]
        seqs = [
            e["collective_seq_id"]
            for e in entries
            if e["profiling_name"] == f"{self.backend_name}:all_reduce"
        ]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), 5)

    def test_legacy_nccl_dump_api(self):
        if self.device_type != "cuda":
            self.skipTest("legacy NCCL dump API is only available in NCCL builds")
        self._init_pg()
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self._wait_for_retired_entries(0, 1)

        generic = json.loads(torch._C._distributed_c10d._dump_fr_trace_json())
        legacy = json.loads(torch._C._distributed_c10d._dump_nccl_trace_json())
        self.assertEqual(generic["entries"], legacy["entries"])
        generic = pickle.loads(torch._C._distributed_c10d._dump_fr_trace())
        legacy = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(generic["entries"], legacy["entries"])


def _make_fr_hook_test_class(backend_name, device_type):
    class FlightRecorderHookTest(AbstractFlightRecorderHookTest, MultiProcessTestCase):
        pass

    FlightRecorderHookTest.backend_name = backend_name
    FlightRecorderHookTest.device_type = device_type
    class_name = "".join(part.capitalize() for part in backend_name.split("-"))
    FlightRecorderHookTest.__name__ = f"{class_name}FlightRecorderHookTest"
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
    test_class = _make_fr_hook_test_class(backend_name, device_type)
    globals()[test_class.__name__] = test_class


if __name__ == "__main__":
    run_tests()

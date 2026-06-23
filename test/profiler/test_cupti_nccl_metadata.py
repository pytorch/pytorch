# Owner(s): ["oncall: profiler"]
"""End-to-end test for the CUPTI-monitor / NCCL-profiler-plugin CollTrace
replacement: a real 2-rank NCCL all_reduce, with the in-process ncclProfiler_v6
plugin producing per-collective metadata and the monitor joining it onto the
collective's GPU kernel.

Requires 2 GPUs + an NCCL build with the v6 profiler plugin (STATIC_PLUGIN). The
monitor leg additionally needs torch and the monitor to share libcupti >= 13.3;
run with that libcupti LD_PRELOAD'd (see the other CUDA monitor tests). When the
monitor cannot subscribe / collects nothing (e.g. libcupti too old), the
metadata-attach test self-skips; the producer test needs no CUPTI at all.
"""

import json
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import run_tests, TestCase


TEST_MULTIGPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2
TEST_NCCL = dist.is_available() and dist.is_nccl_available()
_PORT = "29401"


def _init_pg(rank, world_size, port):
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _producer_worker(rank, world_size, q):
    # Validates the plugin alone (no monitor/CUPTI): NCCL loads ncclProfiler_v6 and,
    # for a real collective, stores the descriptor blob keyed by the pushed id.
    from torch.profiler._cupti.monitor import enable_nccl_metadata_plugin

    enable_nccl_metadata_plugin()  # promote symbol to global scope + STATIC_PLUGIN
    m = torch._C._profiler._cupti_monitor
    _init_pg(rank, world_size, _PORT)
    try:
        t = torch.ones(2048, device=f"cuda:{rank}")
        m.drain_decoded()  # clear residue
        ext_id = 0xC0DE + rank
        m.note_external_push(ext_id)  # the comms-wrapper's per-collective id
        dist.all_reduce(t)
        torch.cuda.synchronize()
        m.note_external_pop()
        _, ext_meta = m.drain_decoded()
        if rank == 0:
            q.put({"blob": ext_meta.get(ext_id), "result": t[0].item()})
    finally:
        dist.destroy_process_group()


def _full_path_worker(rank, world_size, q):
    # Validates the whole chain: plugin blob + monitor decode + the join attaching
    # event["metadata"] onto the real NCCL kernel.
    from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

    from torch.profiler._cupti.cupti_python import CuptiError
    from torch.profiler._cupti.monitor import enable_nccl_metadata_plugin, instance
    from torch.profiler._cupti.observers.profiler import _attach_metadata
    from torch.profiler._cupti.records import Api, ExternalCorrelation, Kernel

    enable_nccl_metadata_plugin()
    monitor = instance()
    captured: dict[int, list] = {}

    def on_columns(cols):
        for k, c in cols.items():
            captured.setdefault(int(k), []).append({int(f): v for f, v in c.items()})

    K = int(ActivityKind.CONCURRENT_KERNEL)
    EC = int(ActivityKind.EXTERNAL_CORRELATION)
    want = {
        ActivityKind.CONCURRENT_KERNEL: {
            Kernel.CORRELATION_ID,
            Kernel.GRAPH_NODE_ID,
            Kernel.START,
            Kernel.END,
            Kernel.NAME,
        },
        ActivityKind.EXTERNAL_CORRELATION: {
            ExternalCorrelation.EXTERNAL_ID,
            ExternalCorrelation.CORRELATION_ID,
            ExternalCorrelation.EXTERNAL_KIND,
        },
        # Both API kinds are needed as the external-correlation carrier: NCCL
        # launches the collective kernel through the driver API.
        ActivityKind.RUNTIME: {Api.CORRELATION_ID},
        ActivityKind.DRIVER: {Api.CORRELATION_ID},
    }
    try:
        obs = monitor.register(want, on_columns)
    except CuptiError as e:
        if rank == 0:
            q.put({"skip": f"monitor subscribe unavailable: {e}"})
        return

    _init_pg(rank, world_size, str(int(_PORT) + 1))
    try:
        t = torch.ones(4096, device=f"cuda:{rank}")
        # One push: the plugin reads it (per-collective key) AND CUPTI emits the
        # EXTERNAL_CORRELATION record linking it to the launched kernel.
        ext_id = monitor.push_external_correlation_id()
        dist.all_reduce(t)
        torch.cuda.synchronize()
        monitor.pop_external_correlation_id()
        monitor.flush(sync=True)
        ext_meta = monitor.take_external_metadata()
        monitor.unregister(obs)

        if rank != 0:
            return

        import numpy as np

        def rows(kind):
            out = []
            for cols in captured.get(kind, []):
                n = len(next(iter(cols.values())))
                for i in range(n):
                    out.append({f: col[i] for f, col in cols.items()})
            return out

        kernel_rows = rows(K)
        ext_rows = rows(EC)
        if not kernel_rows:
            q.put({"skip": "monitor collected no kernels (needs libcupti>=13.3)"})
            return
        names = [r[int(Kernel.NAME)] for r in kernel_rows]
        columns = {
            "kernel": {
                "correlation_id": np.array(
                    [int(r[int(Kernel.CORRELATION_ID)]) for r in kernel_rows],
                    dtype=np.int64,
                ),
                "graph_node_id": np.array(
                    [int(r[int(Kernel.GRAPH_NODE_ID)]) for r in kernel_rows],
                    dtype=np.int64,
                ),
            },
            "external_correlation": {
                "external_id": np.array(
                    [int(r[int(ExternalCorrelation.EXTERNAL_ID)]) for r in ext_rows],
                    dtype=np.int64,
                ),
                "correlation_id": np.array(
                    [int(r[int(ExternalCorrelation.CORRELATION_ID)]) for r in ext_rows],
                    dtype=np.int64,
                ),
            },
        }
        _attach_metadata(columns, ext_meta, None)
        meta = columns["kernel"].get("metadata")
        tagged = [
            json.loads(meta[i])
            for i, name in enumerate(names)
            if meta is not None and meta[i] is not None and "ncclDevKernel" in name
        ]
        q.put({"pushed": ext_id, "blob": ext_meta.get(ext_id), "tagged": tagged})
    finally:
        dist.destroy_process_group()


@unittest.skipIf(not TEST_MULTIGPU, "needs >= 2 GPUs")
@unittest.skipIf(not TEST_NCCL, "needs NCCL")
class TestCuptiNcclMetadata(TestCase):
    def _run(self, worker):
        q = mp.get_context("spawn").SimpleQueue()
        mp.spawn(worker, args=(2, q), nprocs=2, join=True)
        return q.get()

    def test_plugin_produces_blob_under_real_nccl(self):
        # The ncclProfiler_v6 plugin (STATIC_PLUGIN) fires for a real all_reduce and
        # stores the descriptor blob keyed by the pushed external id.
        res = self._run(_producer_worker)
        self.assertEqual(res["result"], 2.0)  # all_reduce over 2 ranks
        self.assertIsNotNone(res["blob"], "plugin did not store a blob")
        meta = json.loads(res["blob"])
        self.assertEqual(meta["func"], "AllReduce")
        self.assertEqual(meta["count"], 2048)
        self.assertEqual(meta["datatype"], "ncclFloat32")

    def test_metadata_attaches_to_collective_kernel(self):
        # Full chain: the blob is joined onto the real NCCL collective kernel via the
        # CUSTOM1 external-correlation record.
        res = self._run(_full_path_worker)
        if "skip" in res:
            self.skipTest(res["skip"])
        self.assertIsNotNone(res["blob"])
        self.assertTrue(
            any(m.get("func") == "AllReduce" for m in res["tagged"]),
            f"no NCCL kernel got AllReduce metadata; tagged={res['tagged']}",
        )


if __name__ == "__main__":
    run_tests()

# Owner(s): ["oncall: distributed"]

import gzip
import json
import os
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed._cuda_graph_annotations import CollectiveAnnotations
from torch.distributed.distributed_c10d import _get_default_group
from torch.profiler import (
    CuspyConfig,
    profile,
    ProfilerActivity,
    ProfilerActivityConfig,
)
from torch.testing._internal.common_cuda import (
    TEST_CUDA_GRAPH_TOOLS_ID,
    TEST_CUPTI_V13_3,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfRocm,
    TemporaryFileName,
)


class TestCollectiveMetadata(MultiProcContinuousTest):
    hw_classification = HardwareClassification.GENERIC

    world_size = 2

    @classmethod
    def backend_str(cls):
        return "gloo"

    def _record(self, fn):
        recorded = []

        def fake_mark_kernels(annotation, *, backward):
            self.assertFalse(backward)
            recorded.append(annotation)
            return mock.MagicMock()

        with mock.patch(
            "torch.distributed._cuda_graph_annotations.mark_kernels",
            fake_mark_kernels,
        ):
            annotations = CollectiveAnnotations()
            try:
                fn()
            finally:
                annotations.close()
        return recorded

    def test_collective_fields(self):
        pg = _get_default_group()
        seq = pg._get_sequence_number_for_group()
        ws = self.world_size

        def run():
            dist.all_reduce(torch.ones(3))
            dist.all_gather_into_tensor(torch.zeros(2 * ws), torch.ones(2))

        allreduce, allgather = self._record(run)
        self.assertEqual(
            allreduce,
            {
                "Collective name": "allreduce",
                "In msg nelems": 3,
                "Out msg nelems": 3,
                "Group size": ws,
                "Process Group Name": pg.group_name,
                "Process Group Description": pg.group_desc,
                "Process Group Ranks": list(range(ws)),
                "dtype": "float32",
                "Seq": seq + 1,
            },
        )
        self.assertEqual(allgather["Collective name"], "_allgather_base")
        self.assertEqual(allgather["In msg nelems"], 2)
        self.assertEqual(allgather["Out msg nelems"], 2 * ws)
        self.assertEqual(allgather["Seq"], seq + 2)
        self.assertEqual(pg._get_sequence_number_for_group(), seq + 2)

    def test_p2p_fields(self):
        peer = (self.rank + 1) % self.world_size

        def run():
            if self.rank % 2 == 0:
                dist.send(torch.ones(2), dst=peer)
                dist.recv(torch.zeros(2), src=peer)
            else:
                dist.recv(torch.zeros(2), src=peer)
                dist.send(torch.ones(2), dst=peer)

        recorded = self._record(run)
        send = next(r for r in recorded if r["Collective name"] == "send")
        recv = next(r for r in recorded if r["Collective name"] == "recv")
        self.assertEqual(send["Dst Rank"], peer)
        self.assertEqual(recv["Src Rank"], peer)
        self.assertNotIn("Seq", send)
        self.assertNotIn("Seq", recv)

    def test_custom_annotate(self):
        recorded = []

        def annotate(metadata):
            recorded.append(metadata["Collective name"])
            return mock.MagicMock()

        annotations = CollectiveAnnotations(annotate)
        try:
            dist.all_reduce(torch.ones(1))
        finally:
            annotations.close()
        self.assertEqual(recorded, ["allreduce"])

    def test_close_unregisters_hooks(self):
        with mock.patch(
            "torch.distributed._cuda_graph_annotations.mark_kernels"
        ) as mark_kernels:
            CollectiveAnnotations().close()
            dist.all_reduce(torch.ones(1))
        mark_kernels.assert_not_called()


class TestCollectiveGraphAnnotations(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _capture(self, **kwargs):
        from torch.cuda.graph_annotations import get_kernel_annotations

        torch.cuda.set_device(self.device)
        x = torch.ones(1024, device=self.device)
        dist.all_reduce(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, enable_annotations=True, **kwargs):
            dist.all_reduce(x)
        return [
            a
            for annotations in get_kernel_annotations().values()
            for a in annotations
            if "Collective name" in a
        ]

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_captured_collective_is_annotated(self):
        if not TEST_CUDA_GRAPH_TOOLS_ID:
            self.skipTest("CUDA graph annotations are unavailable")
        pg = _get_default_group()
        annotations = self._capture()
        self.assertTrue(annotations, "no kernel carries collective metadata")
        for annotation in annotations:
            self.assertEqual(annotation["Collective name"], "allreduce")
            self.assertEqual(annotation["Process Group Name"], pg.group_name)
            self.assertEqual(annotation["Group size"], self.world_size)

    def _record_during_capture(self, **kwargs):
        recorded = []
        with mock.patch(
            "torch.distributed._cuda_graph_annotations.mark_kernels",
            lambda annotation, *, backward: recorded.append(annotation)
            or mock.MagicMock(),
        ):
            self._capture(**kwargs)
            dist.all_reduce(torch.ones(1, device=self.device))
        return recorded

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_hooks_scoped_to_capture(self):
        recorded = self._record_during_capture()
        self.assertEqual([a["Collective name"] for a in recorded], ["allreduce"])
        self.assertEqual(recorded[0]["In msg nelems"], 1024)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_opt_out(self):
        self.assertEqual(
            self._record_during_capture(annotation_config={"collectives": False}), []
        )


class TestCollectiveCuspyAnnotations(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _profile_all_reduce(self, **cuspy_kwargs):
        torch.cuda.set_device(self.device)
        x = torch.ones(1024, device=self.device)
        dist.all_reduce(x)
        torch.cuda.synchronize()
        cuda_config = ProfilerActivityConfig(
            profiler_configs=[CuspyConfig(**cuspy_kwargs)]
        )
        with TemporaryFileName(mode="w+") as trace_path:
            with profile(
                activities=[ProfilerActivity.CPU, {ProfilerActivity.CUDA: cuda_config}]
            ) as prof:
                dist.all_reduce(x)
                torch.cuda.synchronize()
            prof.export_chrome_trace(trace_path)
            gz_path = trace_path + ".gz"
            if os.path.exists(gz_path):
                with gzip.open(gz_path, "rt") as f:
                    events = json.load(f)["traceEvents"]
            else:
                with open(trace_path) as f:
                    events = json.load(f)["traceEvents"]
        return [
            e
            for e in events
            if e.get("cat") == "kernel" and "nccl" in e.get("name", "").lower()
        ]

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_eager_collective_is_annotated(self):
        if not TEST_CUPTI_V13_3:
            self.skipTest("requires libcupti >= 13.3")
        pg = _get_default_group()
        kernels = self._profile_all_reduce()
        self.assertTrue(kernels, "no NCCL kernel in the trace")
        for kernel in kernels:
            self.assertEqual(kernel["args"]["Collective name"], "allreduce")
            self.assertEqual(kernel["args"]["Process Group Name"], pg.group_name)
            self.assertEqual(kernel["args"]["In msg nelems"], 1024)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_opt_out(self):
        if not TEST_CUPTI_V13_3:
            self.skipTest("requires libcupti >= 13.3")
        kernels = self._profile_all_reduce(annotate_collectives=False)
        self.assertTrue(kernels, "no NCCL kernel in the trace")
        for kernel in kernels:
            self.assertNotIn("Collective name", kernel["args"])


if __name__ == "__main__":
    run_tests()

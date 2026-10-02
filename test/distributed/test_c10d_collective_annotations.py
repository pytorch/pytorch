# Owner(s): ["oncall: distributed"]

import contextlib
import gc
import gzip
import json
import os
import weakref
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed._collective_annotations as ca
from torch.distributed._collective_annotations import (
    _format_ranks,
    _GroupHooks,
    _on_cuda,
    CollectiveAnnotations,
)
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
    TestCase,
)


class TestFormatRanks(TestCase):
    def test_matches_record_param_comms(self):
        self.assertEqual(_format_ranks([0, 1]), "[0, 1]")
        self.assertEqual(_format_ranks(list(range(30))), str(list(range(30))))
        self.assertEqual(
            _format_ranks(list(range(40))),
            f"[{', '.join(map(str, range(29)))}, ..., 39]",
        )


_MARK_KERNELS = "torch.cuda._graph_annotations.mark_kernels"


@contextlib.contextmanager
def _as_captured(*, launches=True, capturing=True):
    # Gloo runs collectives on CPU tensors outside a capture, which are skipped.
    with (
        mock.patch.object(ca, "_on_cuda", return_value=launches),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
    ):
        yield


class TestCollectiveMetadata(MultiProcContinuousTest):
    hw_classification = HardwareClassification.GENERIC

    world_size = 2

    @classmethod
    def backend_str(cls):
        return "gloo"

    def _record(self, fn, **filters):
        recorded = []

        def fake_mark_kernels(annotation, *, backward):
            self.assertFalse(backward)
            recorded.append(annotation)
            return mock.MagicMock()

        with (
            _as_captured(**filters),
            mock.patch(
                _MARK_KERNELS,
                fake_mark_kernels,
            ),
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
        self.assertTrue(pg.group_desc)
        self.assertEqual(
            allreduce,
            {
                "Collective name": "all_reduce",
                "In msg nelems": 3,
                "Out msg nelems": 3,
                "Group size": ws,
                "Process Group Name": pg.group_name,
                "Process Group Description": pg.group_desc,
                "Process Group Ranks": str(list(range(ws))),
                "Is asynchronized op": False,
                "Rank": self.rank,
                "dtype": "Float",
                "Seq": seq + 1,
            },
        )
        self.assertEqual(allgather["Collective name"], "all_gather_single")
        self.assertEqual(allgather["In msg nelems"], 2)
        self.assertEqual(allgather["Out msg nelems"], 2 * ws)
        self.assertEqual(allgather["Seq"], seq + 2)
        self.assertEqual(pg._get_sequence_number_for_group(), seq + 2)

    def test_async_and_dtype(self):
        def run():
            dist.all_reduce(torch.ones(1, dtype=torch.bfloat16), async_op=True).wait()

        (allreduce,) = self._record(run)
        self.assertTrue(allreduce["Is asynchronized op"])
        self.assertEqual(allreduce["dtype"], "BFloat16")

    def test_empty_description_omitted(self):
        group = mock.Mock(group_desc="", group_name="pg")
        group.rank.return_value = 0
        with mock.patch.object(dist, "get_process_group_ranks", return_value=[0]):
            fields = ca._GroupFields(group)
        self.assertNotIn("Process Group Description", fields.static)

    def test_skips_outside_capture(self):
        self.assertEqual(
            self._record(lambda: dist.all_reduce(torch.ones(1)), capturing=False), []
        )

    def test_skips_cpu_tensors(self):
        self.assertFalse(_on_cuda(mock.Mock(input_tensors=[torch.ones(1)])))
        self.assertFalse(_on_cuda(mock.Mock(input_tensors=[], output_tensors=[])))
        self.assertEqual(
            self._record(lambda: dist.all_reduce(torch.ones(1)), launches=False), []
        )

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
        self.assertEqual(send["Rank"], peer)
        self.assertEqual(recv["Src Rank"], peer)
        self.assertNotIn("Seq", send)
        self.assertNotIn("Seq", recv)

    def _annotate(self, annotate, fn, *, capturing=True):
        with _as_captured(capturing=capturing):
            annotations = CollectiveAnnotations(annotate)
            try:
                fn()
            finally:
                annotations.close()

    def test_custom_annotate(self):
        recorded = []

        def annotate(metadata):
            recorded.append(metadata["Collective name"])
            return mock.MagicMock()

        self._annotate(annotate, lambda: dist.all_reduce(torch.ones(1)))
        self.assertEqual(recorded, ["all_reduce"])

    def test_custom_annotate_outside_capture(self):
        recorded = []

        def annotate(metadata):
            recorded.append(metadata["Collective name"])
            return mock.MagicMock()

        self._annotate(
            annotate, lambda: dist.all_reduce(torch.ones(1)), capturing=False
        )
        self.assertEqual(recorded, ["all_reduce"])

    def test_same_annotator_entered_once(self):
        recorded = []

        def annotate(metadata):
            recorded.append(metadata["Collective name"])
            return mock.MagicMock()

        with _as_captured():
            outer = CollectiveAnnotations(annotate)
            inner = CollectiveAnnotations(annotate)
            dist.all_reduce(torch.ones(1))
            inner.close()
            dist.all_reduce(torch.ones(1))
            outer.close()
            dist.all_reduce(torch.ones(1))
        self.assertEqual(recorded, ["all_reduce", "all_reduce"])

    def test_annotate_error_does_not_fail_collective(self):
        def annotate(metadata):
            raise RuntimeError("boom")

        x = torch.ones(1)
        self._annotate(annotate, lambda: dist.all_reduce(x))
        self.assertEqual(x.item(), self.world_size)

    def _hooks_with_scope(self, op_id):
        scope = mock.MagicMock()
        hooks = _GroupHooks(_get_default_group())
        with (
            _as_captured(),
            mock.patch.object(ca, "collective_metadata", return_value={}),
            mock.patch.object(ca, "_annotators", (lambda metadata: scope,)),
        ):
            hooks._pre(mock.Mock(op_id=op_id))
        scope.__enter__.assert_called_once()
        return hooks, scope

    def test_post_exits_scope(self):
        hooks, scope = self._hooks_with_scope(op_id=7)
        hooks._post(mock.Mock(op_id=7))
        scope.__exit__.assert_called_once()
        self.assertEqual(hooks._scopes, {})

    def test_new_group_is_hooked(self):
        group = dist.new_group()
        recorded = self._record(lambda: dist.all_reduce(torch.ones(1), group=group))
        self.assertEqual(
            [a["Process Group Name"] for a in recorded], [group.group_name]
        )
        dist.destroy_process_group(group)

    def test_hooks_do_not_keep_group_alive(self):
        group = dist.new_group()
        ref = weakref.ref(group)
        dist.destroy_process_group(group)
        del group
        gc.collect()
        self.assertIsNone(ref())

    def test_disabled_after_close(self):
        with _as_captured(), mock.patch(_MARK_KERNELS) as mark_kernels:
            annotations = CollectiveAnnotations()
            annotations.close()
            annotations.close()
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
            self.assertEqual(annotation["Collective name"], "all_reduce")
            self.assertEqual(annotation["Process Group Name"], pg.group_name)
            self.assertEqual(annotation["Group size"], self.world_size)

    def _record_during_capture(self, **kwargs):
        recorded = []
        with mock.patch(
            _MARK_KERNELS,
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
        self.assertEqual([a["Collective name"] for a in recorded], ["all_reduce"])
        self.assertEqual(recorded[0]["In msg nelems"], 1024)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_opt_out(self):
        self.assertEqual(
            self._record_during_capture(
                annotation_config={"annotate_collectives": False}
            ),
            [],
        )


class TestCollectiveCuspyAnnotations(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls):
        return "nccl"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _profile_all_reduce(self, region=None, **cuspy_kwargs):
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
                if region is None:
                    dist.all_reduce(x)
                else:
                    with torch.profiler.record_function(region):
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
        self._events = events
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
            self.assertEqual(kernel["args"]["Collective name"], "all_reduce")
            self.assertEqual(kernel["args"]["Process Group Name"], pg.group_name)
            self.assertEqual(kernel["args"]["In msg nelems"], 1024)

    @skipIfRocm
    @skip_if_lt_x_gpu(2)
    def test_keeps_enclosing_region(self):
        if not TEST_CUPTI_V13_3:
            self.skipTest("requires libcupti >= 13.3")
        kernels = self._profile_all_reduce(region="outer_region")
        self.assertTrue(kernels, "no NCCL kernel in the trace")
        regions = [
            e
            for e in self._events
            if e.get("cat") == "gpu_user_annotation" and e["name"] == "outer_region"
        ]
        for kernel in kernels:
            self.assertEqual(kernel["args"]["Collective name"], "all_reduce")
            self.assertTrue(
                any(
                    r["tid"] == kernel["tid"]
                    and r["ts"] <= kernel["ts"]
                    and kernel["ts"] + kernel["dur"] <= r["ts"] + r["dur"]
                    for r in regions
                ),
                f"no outer_region annotation covers {kernel}",
            )

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

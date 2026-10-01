# Owner(s): ["oncall: distributed"]

import contextlib
import threading
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed._cuda_graph_annotations as cga
from torch.distributed._cuda_graph_annotations import (
    _format_ranks,
    _GroupHooks,
    _launches_kernels,
    CollectiveAnnotations,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_cuda import TEST_CUDA_GRAPH_TOOLS_ID
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfRocm,
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


@contextlib.contextmanager
def _as_captured(*, launches=True, capturing=True):
    # Gloo runs collectives on CPU tensors outside a capture, which are skipped.
    with (
        mock.patch.object(cga, "_launches_kernels", return_value=launches),
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
                "torch.distributed._cuda_graph_annotations.mark_kernels",
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
                "Collective name": "allreduce",
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
        self.assertEqual(allgather["Collective name"], "_allgather_base")
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
            fields = cga._GroupFields(group)
        self.assertNotIn("Process Group Description", fields.static)

    def test_skips_outside_capture(self):
        self.assertEqual(
            self._record(lambda: dist.all_reduce(torch.ones(1)), capturing=False), []
        )

    def test_skips_cpu_tensors(self):
        self.assertFalse(_launches_kernels(mock.Mock(input_tensors=[torch.ones(1)])))
        self.assertFalse(
            _launches_kernels(mock.Mock(input_tensors=[], output_tensors=[]))
        )
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

    def test_annotate_error_does_not_fail_collective(self):
        with (
            _as_captured(),
            mock.patch(
                "torch.distributed._cuda_graph_annotations.mark_kernels",
                side_effect=RuntimeError("boom"),
            ),
        ):
            annotations = CollectiveAnnotations()
            try:
                x = torch.ones(1)
                dist.all_reduce(x)
            finally:
                annotations.close()
        self.assertEqual(x.item(), self.world_size)

    def _hooks_with_scope(self, op_id):
        scope = mock.MagicMock()
        hooks = _GroupHooks(_get_default_group())
        with (
            _as_captured(),
            mock.patch(
                "torch.distributed._cuda_graph_annotations.collective_metadata",
                return_value={},
            ),
            mock.patch(
                "torch.distributed._cuda_graph_annotations.mark_kernels",
                return_value=scope,
            ),
        ):
            hooks._pre(mock.Mock(op_id=op_id))
        scope.__enter__.assert_called_once()
        return hooks, scope

    def test_close_exits_own_scopes(self):
        # A backend that raises after the pre hook leaves its scope open.
        hooks, scope = self._hooks_with_scope(op_id=7)
        hooks.close()
        scope.__exit__.assert_called_once()
        self.assertEqual(hooks._scopes, {})

    def test_close_leaves_other_thread_scope_to_post(self):
        result = []
        thread = threading.Thread(
            target=lambda: result.append(self._hooks_with_scope(op_id=7))
        )
        thread.start()
        thread.join()
        hooks, scope = result[0]
        group = mock.Mock(wraps=hooks._group)
        hooks._group = group
        hooks.close()
        scope.__exit__.assert_not_called()
        group.unregister_post_hook.assert_not_called()
        hooks._post(mock.Mock(op_id=7))
        scope.__exit__.assert_called_once()
        group.unregister_post_hook.assert_called_once_with(hooks._post_id)

    def test_pre_after_close_exits_scope(self):
        scope = mock.MagicMock()
        hooks = _GroupHooks(_get_default_group())
        hooks.close()
        with (
            _as_captured(),
            mock.patch(
                "torch.distributed._cuda_graph_annotations.collective_metadata",
                return_value={},
            ),
            mock.patch(
                "torch.distributed._cuda_graph_annotations.mark_kernels",
                return_value=scope,
            ),
        ):
            hooks._pre(mock.Mock(op_id=7))
        scope.__exit__.assert_called_once()
        self.assertEqual(hooks._scopes, {})

    def test_close_unregisters_hooks(self):
        with (
            _as_captured(),
            mock.patch(
                "torch.distributed._cuda_graph_annotations.mark_kernels"
            ) as mark_kernels,
        ):
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


if __name__ == "__main__":
    run_tests()

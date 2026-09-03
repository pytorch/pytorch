# Owner(s): ["oncall: distributed"]

import copy
import math
from typing import Any

import torch
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.config_manager import JobConfig
from torch.distributed.flight_recorder.components.types import (
    COLLECTIVES,
    MatchInfo,
    MatchState,
    Op,
)
from torch.distributed.flight_recorder.components.utils import match_one_event
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = torch.distributed.get_default_backend_for_device(device_type)


def create_one_event(
    collective_name,
    pg_info,
    input_sizes,
    output_sizes,
    state="scheduled",
    collective_seq_id=0,
    p2p_seq_id=0,
    output_dtypes="float32",
    input_dtypes="float32",
    backend="nccl",
):
    return {
        "profiling_name": f"{backend}:{collective_name}",
        "state": state,
        "process_group": pg_info,
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        "input_dtypes": input_dtypes,
        "output_dtypes": output_dtypes,
        "collective_seq_id": str(collective_seq_id),
        "p2p_seq_id": str(p2p_seq_id),
        "time_created_ns": 0,
        "frames": [],
    }


class FlightRecorderEventTest(TestCase):
    def test_match_one_event(self):
        e1 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        membership = {"0": {0, 1}}
        self.assertEqual(
            match_one_event(e1, e1, membership, "0").state, MatchState.FULLY_MATCHED
        )

        e2 = create_one_event(
            "all_gather", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        self.assertEqual(
            match_one_event(e1, e2, membership, "0").state,
            MatchState.COLLECTIVE_TYPE_MISMATCH,
        )

        e3 = create_one_event(
            "all_to_all", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        e4 = create_one_event(
            "all_to_all", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        self.assertEqual(
            match_one_event(e3, e4, membership, "0").state, MatchState.UNDECIDED
        )

        e5 = create_one_event(
            "all_reduce", ("0", "default"), [[5, 4]], [[4, 4]], "scheduled", 1, 1
        )
        self.assertEqual(
            match_one_event(e1, e5, membership, "0").state,
            MatchState.SIZE_OR_SYNTAX_MISMATCH,
        )

        e6 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[5, 4]], "scheduled", 1, 2
        )
        self.assertEqual(
            match_one_event(e1, e6, membership, "0").state,
            MatchState.SIZE_OR_SYNTAX_MISMATCH,
        )

        e7 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[5, 4]], "scheduled", 2
        )
        self.assertEqual(
            match_one_event(e7, e7, membership, "0").state,
            MatchState.SIZE_OR_SYNTAX_MISMATCH,
        )

        e9 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[4, 4]], "completed", 1
        )
        self.assertEqual(
            match_one_event(e1, e9, membership, "0").state,
            MatchState.COLLECTIVE_STATE_MISMATCH,
        )

        e10 = create_one_event(
            "all_reduce",
            ("0", "default"),
            [[4, 4]],
            [[4, 4]],
            "completed",
            1,
            output_dtypes="float16",
        )
        self.assertEqual(
            match_one_event(e10, e9, membership, "0").state,
            MatchState.COLLECTIVE_DTYPE_MISMATCH,
        )

        e11 = create_one_event(
            "gather",
            ("0", "default"),
            [[4, 4]],
            [[4, 4], [4, 4]],
            "completed",
            1,
            output_dtypes="float32",
        )
        e12 = create_one_event(
            "gather",
            ("0", "default"),
            [[4, 4]],
            [[]],
            "completed",
            1,
            output_dtypes="",
        )
        self.assertEqual(
            match_one_event(e11, e12, membership, "0").state,
            MatchState.FULLY_MATCHED,
        )
        e13 = create_one_event(
            "gather",
            ("0", "default"),
            [[4, 4]],
            [[4, 4]],
            "completed",
            1,
            output_dtypes="",
        )
        self.assertEqual(
            match_one_event(e11, e13, membership, "0").state,
            MatchState.FULLY_MATCHED,
        )

    def test_all_events(self):
        for collective in sorted(COLLECTIVES):
            input_sizes = [[4, 4]]
            output_sizes = [[4, 4]]
            expectedState = MatchState.FULLY_MATCHED
            if collective in [
                "reduce_scatter",
                "_reduce_scatter_base",
                "reduce_scatter_tensor_coalesced",
            ]:
                input_sizes = [[4, 4]]
                output_sizes = [[input_sizes[0][0] * 2]]
            if collective in [
                "all_gather",
                "_all_gather_base",
                "all_gather_into_tensor_coalesced",
            ]:
                output_sizes = [[math.prod(input_sizes[0]) * 2]]
            if collective in ["all_to_all", "all_to_all_single"]:
                expectedState = MatchState.UNDECIDED
            event = create_one_event(
                collective, ("0", "default"), input_sizes, output_sizes, "scheduled", 1
            )
            membership = {"0": {0, 1}}
            result = match_one_event(event, event, membership, "0").state
            self.assertEqual(result, expectedState)

    def test_collective_variants_are_distinct_types(self):
        # c10d's FlightRecorderHook used to record every allgather variant as
        # "all_gather" and both alltoalls as "all_to_all", so a rank calling
        # one could match a peer calling the other.
        membership = {"0": {0, 1}}
        for a, b in (
            ("all_gather", "_all_gather_base"),
            ("all_reduce", "allreduce_coalesced"),
            ("reduce_scatter", "_reduce_scatter_base"),
            ("all_to_all", "all_to_all_single"),
        ):
            ea = create_one_event(a, ("0", "default"), [[8]], [[8]], "scheduled", 1)
            eb = create_one_event(b, ("0", "default"), [[8]], [[8]], "scheduled", 1)
            self.assertEqual(
                match_one_event(ea, eb, membership, "0").state,
                MatchState.COLLECTIVE_TYPE_MISMATCH,
                msg=f"{a} vs {b}",
            )

    def test_all_gather_base_numel_is_checked(self):
        # The rule was spelled "all_gather_base"; no producer has ever written
        # that, so the branch never ran and a flattened all-gather with the
        # wrong output numel passed.
        membership = {"0": {0, 1}}
        good = create_one_event(
            "_all_gather_base", ("0", "default"), [[8]], [[16]], "scheduled", 1
        )
        bad = create_one_event(
            "_all_gather_base", ("0", "default"), [[8]], [[24]], "scheduled", 1
        )
        self.assertEqual(
            match_one_event(good, good, membership, "0").state,
            MatchState.FULLY_MATCHED,
        )
        self.assertEqual(
            match_one_event(bad, bad, membership, "0").state,
            MatchState.SIZE_OR_SYNTAX_MISMATCH,
        )

    def test_all_to_all_single_uneven_splits_are_not_a_mismatch(self):
        # alltoall_base with uneven splits gives each rank a different buffer
        # size, so only the summed numel across the group can be checked --
        # the same treatment "all_to_all" gets, which is why it is not folded
        # into the per-rank size comparison.
        membership = {"0": {0, 1}}
        e0 = create_one_event(
            "all_to_all_single", ("0", "default"), [[8]], [[12]], "scheduled", 1
        )
        e1 = create_one_event(
            "all_to_all_single", ("0", "default"), [[12]], [[8]], "scheduled", 1
        )
        self.assertEqual(
            match_one_event(e0, e1, membership, "0").state, MatchState.UNDECIDED
        )

    def test_recv_any_source_has_no_peer(self):
        # recv_any_source names no peer. Reading it as rank -1 made
        # _init_global_src_dst index the group's rank list from the end and
        # pin the recv on the highest-ranked member.
        membership = {"0": {2, 5}}
        recv = create_one_event(
            "recv 1<-?", ("0", "default"), [[8]], [[8]], "scheduled", 0, 1
        )
        op = Op(recv, membership, "0")
        self.assertIsNone(op.src)
        self.assertIsNone(op._src_g)
        self.assertEqual((op.dst, op._dst_g), (1, 5))
        # ... and it matches whichever rank actually sent to it.
        send = create_one_event(
            "send 0->1", ("0", "default"), [[8]], [[8]], "scheduled", 0, 1
        )
        self.assertEqual(
            match_one_event(send, recv, membership, "0").state,
            MatchState.FULLY_MATCHED,
        )
        self.assertEqual(
            match_one_event(recv, send, membership, "0").state,
            MatchState.FULLY_MATCHED,
        )
        # A genuine size mismatch is still caught.
        short = create_one_event(
            "send 0->1", ("0", "default"), [[4]], [[4]], "scheduled", 0, 1
        )
        self.assertEqual(
            match_one_event(short, recv, membership, "0").state,
            MatchState.SIZE_OR_SYNTAX_MISMATCH,
        )


class FlightRecorderOpBackendTest(TestCase):
    """Tests that the Op class accepts all supported backend prefixes."""

    def _make_event(self, backend: str, collective: str = "all_reduce"):
        return {
            "profiling_name": f"{backend}:{collective}",
            "state": "completed",
            "process_group": ("0", "default"),
            "input_sizes": [[4, 4]],
            "output_sizes": [[4, 4]],
            "input_dtypes": "float32",
            "output_dtypes": "float32",
            "collective_seq_id": "1",
            "p2p_seq_id": "0",
            "time_created_ns": 0,
            "frames": [],
        }

    def test_nccl_backend(self):
        op = Op(self._make_event("nccl"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_ncclx_backend(self):
        op = Op(self._make_event("ncclx"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_gloo_backend(self):
        op = Op(self._make_event("gloo"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_xccl_backend(self):
        op = Op(self._make_event("xccl"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_nccl2_backend(self):
        # c10d's FlightRecorderHook writes the backend's own name, so a group
        # running nccl2 produces "nccl2:<op>".
        op = Op(self._make_event("nccl2"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_c10d_backend(self):
        # Fallback name the hook writes when it cannot identify the backend.
        op = Op(self._make_event("c10d"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_nccl2_backend_p2p(self):
        send = self._make_event("nccl2", "send 0->1")
        op = Op(send, {"0": {0, 1}}, "0")
        self.assertEqual((op.type, op.src, op.dst), ("send", 0, 1))
        recv = self._make_event("nccl2", "recv 1<-0")
        op = Op(recv, {"0": {0, 1}}, "0")
        self.assertEqual((op.type, op.src, op.dst), ("recv", 0, 1))

    def test_unsupported_backend_is_not_fatal(self):
        # The hook attaches to any backend and records under whatever name it
        # reports, so an unknown comm library must not cost us the collective.
        op = Op(self._make_event("unknown_backend"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_backend_name_with_colon(self):
        # A stray colon in the comm library name must not eat the op name.
        op = Op(self._make_event("weird:backend"), {"0": {0, 1}}, "0")
        self.assertEqual(op.type, "all_reduce")

    def test_missing_colon_raises(self):
        with self.assertRaises(AssertionError):
            Op(
                {**self._make_event("nccl"), "profiling_name": "all_reduce"},
                {"0": {0, 1}},
                "0",
            )


class FlightMatchInfoTest(TestCase):
    def test_match_info(self):
        m1 = MatchInfo(MatchState.FULLY_MATCHED, "rank 0")
        m2 = MatchInfo(MatchState.FULLY_MATCHED, "rank 1")
        self.assertEqual(m1.state, MatchState.FULLY_MATCHED)
        self.assertEqual(m1.state, m2.state)
        self.assertEqual(str(m1), "Error type: FULLY_MATCHED, rank 0")
        self.assertEqual(str(m2), "Error type: FULLY_MATCHED, rank 1")


LOADED_FR_DETAIL_TEMPLATE: dict[str, dict[str, Any]] = {
    "dump_file_rank_0": {
        "entries": [],
        "pg_config": {
            "0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"},
            "1": {"name": "1", "desc": "sub_pg", "ranks": "[0]"},
        },
        "rank": 0,
    },
    "dump_file_rank_1": {
        "entries": [],
        "pg_config": {
            "0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"},
            "1": {"name": "1", "desc": "sub_pg", "ranks": "[1]"},
        },
        "rank": 1,
    },
}


def create_one_entry(
    record_id,
    collective_name,
    input_sizes,
    output_sizes,
    state="completed",
    collective_seq_id=0,
    p2p_seq_id=0,
    output_dtypes="float32",
    pg_info=("0", "default"),
    input_dtypes="float32",
    backend="nccl",
):
    event = create_one_event(
        collective_name,
        pg_info,
        input_sizes,
        output_sizes,
        state,
        collective_seq_id,
        p2p_seq_id,
        output_dtypes,
        input_dtypes,
        backend,
    )
    event.update({"record_id": record_id})
    event.update({"is_p2p": False})
    return event


class FlightRecorderE2ETest(TestCase):
    def testBuildDB(self):
        config = JobConfig()
        args = config.parse_args([])
        version = "2.8"  # Same as the version in FlightRecorder.hpp
        LOADED_FR_DETAIL_TEMPLATE["dump_file_rank_0"]["version"] = version
        LOADED_FR_DETAIL_TEMPLATE["dump_file_rank_1"]["version"] = version
        # Test case 1: matched all_reduce case.
        details1 = copy.deepcopy(LOADED_FR_DETAIL_TEMPLATE)
        details1["dump_file_rank_0"]["entries"].append(
            create_one_entry(0, "all_reduce", [[4, 4]], [[4, 4]])
        )
        details1["dump_file_rank_1"]["entries"].append(
            create_one_entry(0, "all_reduce", [[4, 4]], [[4, 4]])
        )
        details1["dump_file_rank_0"]["entries"].append(
            create_one_entry(
                1, "all_reduce", [[5, 5]], [[5, 5]], pg_info=("1", "sub_pg")
            )
        )
        details1["dump_file_rank_1"]["entries"].append(
            create_one_entry(
                1, "all_reduce", [[5, 5]], [[5, 5]], pg_info=("1", "sub_pg")
            )
        )
        db = build_db(details1, args, version)
        self.assertEqual(len(db.collectives), 3)
        self.assertEqual(db.collectives[0].record_id, 0)
        self.assertEqual(db.collectives[0].collective_name, f"{backend}:all_reduce")
        self.assertEqual(db.collectives[0].pass_check, True)
        self.assertEqual(db.collectives[1].record_id, 1)
        self.assertEqual(db.collectives[1].collective_name, f"{backend}:all_reduce")
        self.assertEqual(db.collectives[1].pass_check, True)
        self.assertEqual(db.collectives[2].pass_check, True)
        # Test case 2: matched allreduce_coalesced case.
        details2 = copy.deepcopy(LOADED_FR_DETAIL_TEMPLATE)
        details2["dump_file_rank_0"]["entries"].append(
            create_one_entry(0, "allreduce_coalesced", [[4, 4]], [[4, 4]])
        )
        details2["dump_file_rank_1"]["entries"].append(
            create_one_entry(0, "allreduce_coalesced", [[4, 4]], [[4, 4]])
        )
        db = build_db(details2, args, version)
        self.assertEqual(len(db.collectives), 1)
        self.assertEqual(db.collectives[0].record_id, 0)
        self.assertEqual(
            db.collectives[0].collective_name, f"{backend}:allreduce_coalesced"
        )
        self.assertEqual(db.collectives[0].pass_check, True)
        # Test case 3: matched slow path, two broadcast coalesce case.
        details3 = copy.deepcopy(LOADED_FR_DETAIL_TEMPLATE)
        # sequence ID should not increase for coalesced collectives
        details3["dump_file_rank_0"]["entries"].append(
            create_one_entry(0, "broadcast", [[4, 4]], [[4, 4]])
        )
        details3["dump_file_rank_0"]["entries"].append(
            create_one_entry(1, "broadcast", [[4, 4]], [[4, 4]])
        )
        details3["dump_file_rank_0"]["entries"].append(
            create_one_entry(2, "coalesced", [[]], [[]])
        )
        details3["dump_file_rank_1"]["entries"].append(
            create_one_entry(0, "broadcast", [[4, 4]], [[4, 4]])
        )
        details3["dump_file_rank_1"]["entries"].append(
            create_one_entry(1, "broadcast", [[4, 4]], [[4, 4]])
        )
        details3["dump_file_rank_1"]["entries"].append(
            create_one_entry(2, "coalesced", [[]], [[]])
        )
        db = build_db(details3, args, version)
        self.assertEqual(len(db.collectives), 1)
        self.assertEqual(db.collectives[0].record_id, 2)
        self.assertEqual(db.collectives[0].collective_name, f"{backend}:coalesced")
        self.assertEqual(db.collectives[0].pass_check, True)
        # Test case 4: mis-matched uneven all-gather case.
        details4 = copy.deepcopy(LOADED_FR_DETAIL_TEMPLATE)
        # sequence ID should not increase for coalesced collectives
        details4["dump_file_rank_0"]["entries"].append(
            create_one_entry(0, "_broadcast_oop", [[4, 4]], [[4, 4]])
        )
        details4["dump_file_rank_0"]["entries"].append(
            create_one_entry(1, "_broadcast_oop", [[5, 5]], [[5, 5]])
        )
        details4["dump_file_rank_0"]["entries"].append(
            create_one_entry(2, "ALLGATHER_coalesced", [[]], [[]])
        )
        details4["dump_file_rank_1"]["entries"].append(
            create_one_entry(0, "_broadcast_oop", [[4, 4]], [[4, 4]])
        )
        details4["dump_file_rank_1"]["entries"].append(
            create_one_entry(1, "_broadcast_oop", [[4, 4]], [[4, 4]])
        )
        details4["dump_file_rank_1"]["entries"].append(
            create_one_entry(2, "ALLGATHER_coalesced", [[]], [[]])
        )
        db = build_db(details4, args, version)
        self.assertEqual(len(db.collectives), 1)
        self.assertEqual(db.collectives[0].record_id, 1)
        self.assertEqual(db.collectives[0].collective_name, f"{backend}:_broadcast_oop")
        self.assertEqual(db.collectives[0].pass_check, False)
        # Test case 5: matched uneven reduce scatter case.
        details5 = copy.deepcopy(LOADED_FR_DETAIL_TEMPLATE)
        # sequence ID should not increase for coalesced collectives
        details5["dump_file_rank_0"]["entries"].append(
            create_one_entry(0, "_reduce_oop", [[4, 4]], [[4, 4]])
        )
        details5["dump_file_rank_0"]["entries"].append(
            create_one_entry(1, "_reduce_oop", [[4, 4]], [[4, 4]])
        )
        details5["dump_file_rank_0"]["entries"].append(
            create_one_entry(2, "REDUCE_SCATTER_coalesced", [[]], [[]])
        )
        details5["dump_file_rank_1"]["entries"].append(
            create_one_entry(0, "_reduce_oop", [[4, 4]], [[4, 4]])
        )
        details5["dump_file_rank_1"]["entries"].append(
            create_one_entry(1, "_reduce_oop", [[4, 4]], [[4, 4]])
        )
        details5["dump_file_rank_1"]["entries"].append(
            create_one_entry(2, "REDUCE_SCATTER_coalesced", [[]], [[]])
        )
        db = build_db(details5, args, version)
        self.assertEqual(len(db.collectives), 1)
        self.assertEqual(db.collectives[0].record_id, 2)
        self.assertEqual(
            db.collectives[0].collective_name, f"{backend}:REDUCE_SCATTER_coalesced"
        )
        self.assertEqual(db.collectives[0].pass_check, True)
        # Test case 6: empty coalesced call on rank 0 case.
        details6 = copy.deepcopy(LOADED_FR_DETAIL_TEMPLATE)
        # sequence ID should not increase for coalesced collectives
        details6["dump_file_rank_0"]["entries"].append(
            create_one_entry(0, "all_reduce", [[4, 4]], [[4, 4]])
        )
        details6["dump_file_rank_1"]["entries"].append(
            create_one_entry(0, "all_reduce", [[4, 4]], [[4, 4]])
        )
        details6["dump_file_rank_1"]["entries"].append(
            create_one_entry(1, "_reduce_oop", [[4, 4]], [[4, 4]])
        )
        details6["dump_file_rank_1"]["entries"].append(
            create_one_entry(2, "_reduce_oop", [[4, 4]], [[4, 4]])
        )
        details6["dump_file_rank_1"]["entries"].append(
            create_one_entry(3, "REDUCE_SCATTER_coalesced", [[]], [[]])
        )
        db = build_db(details6, args, version)
        self.assertEqual(len(db.collectives), 2)
        self.assertEqual(db.collectives[1].collective_name, f"{backend}:_reduce_oop")
        self.assertEqual(db.collectives[1].record_id, 1)
        self.assertEqual(db.collectives[1].pass_check, True)


class FlightRecorderHookShapeTest(TestCase):
    """Analysis of traces written by c10d's FlightRecorderHook.

    The hook records the tensors the dispatcher hands it. For the in-place
    collectives that is the same buffer on both sides, and for the list-form
    all_gather / reduce_scatter it is one shard-shaped buffer per rank rather
    than the single flattened buffer a native backend records. Both spellings
    have to analyze with no mismatch, or a real desync ends up buried under
    fabricated ones.
    """

    version = "2.8"  # Same as the version in FlightRecorder.hpp

    def _entry(self, record_id, collective_name, input_sizes, output_sizes, seq):
        return create_one_entry(
            record_id,
            collective_name,
            input_sizes,
            output_sizes,
            state="scheduled",
            collective_seq_id=seq,
            input_dtypes=["Float"] * len(input_sizes),
            output_dtypes=["Float"] * len(output_sizes),
            backend="nccl2",
        )

    def _build_db(self, entries_per_rank):
        details = copy.deepcopy(LOADED_FR_DETAIL_TEMPLATE)
        for rank, entries in enumerate(entries_per_rank):
            details[f"dump_file_rank_{rank}"]["version"] = self.version
            details[f"dump_file_rank_{rank}"]["entries"] = entries
        return build_db(details, JobConfig().parse_args([]), self.version)

    def _build_db_symmetric(self, entries):
        # Both ranks recorded the same thing, so anything build_db reports is
        # the analyzer misreading the shape, not a desync.
        return self._build_db([copy.deepcopy(entries), copy.deepcopy(entries)])

    def _match(self, entry):
        membership = {"0": {0, 1}}
        return match_one_event(entry, copy.deepcopy(entry), membership, "0").state

    def test_matcher_accepts_hook_shapes(self):
        for name, input_sizes, output_sizes in (
            ("all_reduce", [[8]], [[8]]),
            ("allreduce_coalesced", [[8], [4]], [[8], [4]]),
            ("reduce", [[8]], [[8]]),
            ("all_gather", [[8]], [[8], [8]]),
            ("reduce_scatter", [[8], [8]], [[8]]),
        ):
            entry = self._entry(0, name, input_sizes, output_sizes, 1)
            self.assertEqual(self._match(entry), MatchState.FULLY_MATCHED, msg=name)

    def test_hook_shaped_run_has_no_mismatch(self):
        db = self._build_db_symmetric(
            [
                self._entry(0, "all_reduce", [[8]], [[8]], 1),
                self._entry(1, "reduce", [[8]], [[8]], 2),
                self._entry(2, "all_gather", [[8]], [[8], [8]], 3),
                self._entry(3, "reduce_scatter", [[8], [8]], [[8]], 4),
                self._entry(4, "allreduce_coalesced", [[8], [4]], [[8], [4]], 5),
            ]
        )
        self.assertEqual(
            [c.collective_name for c in db.collectives],
            [
                "nccl2:all_reduce",
                "nccl2:reduce",
                "nccl2:all_gather",
                "nccl2:reduce_scatter",
                "nccl2:allreduce_coalesced",
            ],
        )
        self.assertEqual([c.pass_check for c in db.collectives], [True] * 5)

    def test_list_form_all_gather_mismatch_is_caught(self):
        # The list form must not become a blanket accept: one buffer per rank
        # is the whole point, so a list of the wrong length is still wrong.
        db = self._build_db_symmetric(
            [self._entry(0, "all_gather", [[8]], [[8], [8], [8]], 1)]
        )
        self.assertEqual([c.pass_check for c in db.collectives], [False])

    def test_list_form_reduce_scatter_mismatch_is_caught(self):
        db = self._build_db_symmetric(
            [self._entry(0, "reduce_scatter", [[8], [8], [8]], [[8]], 1)]
        )
        self.assertEqual([c.pass_check for c in db.collectives], [False])

    def test_flattened_form_mismatch_is_still_caught(self):
        # Native shapes keep their old verdict.
        for name, input_sizes, output_sizes in (
            ("all_gather", [[8]], [[17]]),
            ("reduce_scatter", [[17]], [[8]]),
        ):
            db = self._build_db_symmetric(
                [self._entry(0, name, input_sizes, output_sizes, 1)]
            )
            self.assertEqual([c.pass_check for c in db.collectives], [False], msg=name)


if __name__ == "__main__":
    run_tests()

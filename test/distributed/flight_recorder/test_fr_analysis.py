# Owner(s): ["oncall: distributed"]

import copy
import math
import pathlib
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from tools.flight_recorder.components.builder import build_db
from tools.flight_recorder.components.config_manager import JobConfig
from tools.flight_recorder.components.types import COLLECTIVES, MatchInfo, MatchState
from tools.flight_recorder.components.utils import match_one_event


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))

from torch.testing._internal.common_utils import run_tests, TestCase


def create_one_event(
    collective_name,
    pg_info,
    input_sizes,
    output_sizes,
    state="scheduled",
    collective_seq_id=0,
    p2p_seq_id=0,
    output_dtypes="float32",
):
    return {
        "profiling_name": f"nccl:{collective_name}",
        "state": state,
        "process_group": pg_info,
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        "input_dtypes": "float32",
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
            if collective == "all_to_all":
                expectedState = MatchState.UNDECIDED
            event = create_one_event(
                collective, ("0", "default"), input_sizes, output_sizes, "scheduled", 1
            )
            membership = {"0": {0, 1}}
            result = match_one_event(event, event, membership, "0").state
            self.assertEqual(result, expectedState)


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
        self.assertEqual(db.collectives[0].collective_name, "nccl:all_reduce")
        self.assertEqual(db.collectives[0].pass_check, True)
        self.assertEqual(db.collectives[1].record_id, 1)
        self.assertEqual(db.collectives[1].collective_name, "nccl:all_reduce")
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
        self.assertEqual(db.collectives[0].collective_name, "nccl:allreduce_coalesced")
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
        self.assertEqual(db.collectives[0].collective_name, "nccl:coalesced")
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
        self.assertEqual(db.collectives[0].collective_name, "nccl:_broadcast_oop")
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
            db.collectives[0].collective_name, "nccl:REDUCE_SCATTER_coalesced"
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
        self.assertEqual(db.collectives[1].collective_name, "nccl:_reduce_oop")
        self.assertEqual(db.collectives[1].record_id, 1)
        self.assertEqual(db.collectives[1].pass_check, True)


if __name__ == "__main__":
    run_tests()

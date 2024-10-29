# Owner(s): ["oncall: distributed"]

import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from tools.flight_recorder.components.types import MatchState
from tools.flight_recorder.components.utils import match_one_event


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))

from torch.testing._internal.common_utils import run_tests, TestCase


def create_one_event(
    collectcive_name,
    pg_info,
    input_sizes,
    output_sizes,
    state="scheduled",
    collective_seq_id=0,
    p2p_seq_id=0,
    output_dtypes="float32",
):
    return {
        "profiling_name": f"nccl:{collectcive_name}",
        "state": state,
        "process_group": pg_info,
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        "input_dtypes": "float32",
        "output_dtypes": output_dtypes,
        "collective_seq_id": str(collective_seq_id),
        "p2p_seq_id": str(p2p_seq_id),
    }


class FlightRecorderEventTest(TestCase):
    def test_match_one_event(self):
        e1 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        membership = {"0": {0, 1}}
        self.assertEqual(
            match_one_event(e1, e1, membership, "0"), MatchState.FULLY_MATCHED
        )

        e2 = create_one_event(
            "all_gather", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        self.assertEqual(
            match_one_event(e1, e2, membership, "0"),
            MatchState.COLLECTIVE_TYPE_MISMATCH,
        )

        e3 = create_one_event(
            "all_to_all", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        e4 = create_one_event(
            "all_to_all", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        self.assertEqual(match_one_event(e3, e4, membership, "0"), MatchState.UNDECIDED)

        e5 = create_one_event(
            "all_reduce", ("0", "default"), [[5, 4]], [[4, 4]], "scheduled", 1, 1
        )
        self.assertEqual(
            match_one_event(e1, e5, membership, "0"), MatchState.SIZE_OR_SYNTAX_MISMATCH
        )

        e6 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[5, 4]], "scheduled", 1, 2
        )
        self.assertEqual(
            match_one_event(e1, e6, membership, "0"), MatchState.SIZE_OR_SYNTAX_MISMATCH
        )

        e7 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[5, 4]], "scheduled", 2
        )
        self.assertEqual(
            match_one_event(e7, e7, membership, "0"), MatchState.SIZE_OR_SYNTAX_MISMATCH
        )

        e9 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[4, 4]], "completed", 1
        )
        self.assertEqual(
            match_one_event(e1, e9, membership, "0"),
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
            match_one_event(e10, e9, membership, "0"),
            MatchState.COLLECTIVE_DTYPE_MISMATCH,
        )


if __name__ == "__main__":
    run_tests()

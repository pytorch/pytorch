# Owner(s): ["oncall: distributed"]


from tools.flight_recorder.fr_trace import match_one_event, MatchState

from torch.testing._internal.common_utils import run_tests, TestCase


def create_one_event(
    collectcive_name,
    pg_info,
    input_sizes,
    output_sizes,
    state="scheduled",
    collective_seq_id=0,
    p2p_seq_id=0,
):
    return {
        "profiling_name": f"nccl:{collectcive_name}",
        "state": state,
        "process_group": pg_info,
        "input_sizes": input_sizes,
        "output_sizes": output_sizes,
        "collective_seq_id": str(collective_seq_id),
        "p2p_seq_id": str(p2p_seq_id),
    }


class FlightRecorderEventTest(TestCase):
    def test_match_one_event(self):
        e1 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        membership = {"0": {0, 1}}
        self.assertEqual(match_one_event(e1, e1, membership), MatchState.FULLY_MATCHED)

        e2 = create_one_event(
            "all_gather", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        self.assertEqual(
            match_one_event(e1, e2, membership), MatchState.COLLECTIVE_TYPE_MISMATCH
        )

        e3 = create_one_event(
            "alltoall", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        e4 = create_one_event(
            "alltoall", ("0", "default"), [[4, 4]], [[4, 4]], "scheduled", 1
        )
        self.assertEqual(match_one_event(e3, e4, membership), MatchState.UNDECIDED)

        e5 = create_one_event(
            "all_reduce", ("0", "default"), [[5, 4]], [[4, 4]], "scheduled", 1, 1
        )
        self.assertEqual(
            match_one_event(e1, e5, membership), MatchState.SIZE_OR_SYNTAX_MISMATCH
        )

        e6 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[5, 4]], "scheduled", 1, 2
        )
        self.assertEqual(
            match_one_event(e1, e6, membership), MatchState.SIZE_OR_SYNTAX_MISMATCH
        )

        e7 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[5, 4]], "scheduled", 2
        )
        self.assertEqual(
            match_one_event(e7, e7, membership), MatchState.SIZE_OR_SYNTAX_MISMATCH
        )

        e9 = create_one_event(
            "all_reduce", ("0", "default"), [[4, 4]], [[4, 4]], "completed", 1
        )
        self.assertEqual(
            match_one_event(e1, e9, membership), MatchState.COLLECTIVE_STATE_MISMATCH
        )


if __name__ == "__main__":
    run_tests()

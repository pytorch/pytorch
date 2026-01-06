# Owner(s): ["oncall: distributed checkpointing"]


from torch.distributed.checkpoint._experimental.types import RankInfo, STATE_DICT
from torch.testing._internal.common_utils import run_tests, TestCase


class TestRankInfo(TestCase):
    def test_rank_info_initialization(self):
        """Test that RankInfo initializes correctly with all parameters."""
        # Create a RankInfo instance with all parameters
        rank_info = RankInfo(
            global_rank=0,
            global_world_size=4,
        )

        # Verify that all attributes are set correctly
        self.assertEqual(rank_info.global_rank, 0)
        self.assertEqual(rank_info.global_world_size, 4)

    def test_rank_info_default_initialization(self):
        """Test that RankInfo initializes correctly with default parameters."""
        # Create a RankInfo instance with only required parameters
        rank_info = RankInfo(
            global_rank=0,
            global_world_size=1,
        )

        # Verify that all attributes are set correctly
        self.assertEqual(rank_info.global_rank, 0)
        self.assertEqual(rank_info.global_world_size, 1)

    def test_state_dict_type_alias(self):
        """Test that STATE_DICT type alias works correctly."""
        # Create a state dictionary
        state_dict = {"model": {"weight": [1, 2, 3]}, "optimizer": {"lr": 0.01}}

        # Verify that it can be assigned to a variable of type STATE_DICT
        state_dict_var: STATE_DICT = state_dict
        self.assertEqual(state_dict_var, state_dict)


if __name__ == "__main__":
    run_tests()

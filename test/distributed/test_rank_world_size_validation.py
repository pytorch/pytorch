# Owner(s): ["oncall: distributed"]

from torch.distributed.distributed_c10d import _check_rank_world_size_values
from torch.testing._internal.common_utils import run_tests, TestCase


class RankWorldSizeValidationTest(TestCase):
    def test_accepts_valid_values(self) -> None:
        _check_rank_world_size_values(rank=0, world_size=1)
        _check_rank_world_size_values(rank=2, world_size=4, local_rank=1)

    def test_negative_rank_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank must be non-negative"):
            _check_rank_world_size_values(rank=-1, world_size=4)

    def test_world_size_less_than_one_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "world_size must be at least 1",
        ):
            _check_rank_world_size_values(rank=0, world_size=0)

    def test_rank_exceeds_world_size_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "rank \\(3\\) must be less than world_size",
        ):
            _check_rank_world_size_values(rank=3, world_size=3)

    def test_negative_local_rank_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "local_rank must be non-negative",
        ):
            _check_rank_world_size_values(
                rank=0,
                world_size=2,
                local_rank=-1,
            )


if __name__ == "__main__":
    run_tests()

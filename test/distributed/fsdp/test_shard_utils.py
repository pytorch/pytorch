from torch.distributed.fsdp.shard_utils import (
    _offsets_to_split_sizes,
)
from torch.testing._internal.common_utils import (
    TestCase,
)

class TestUtils(TestCase):
    def test_offsets_to_split_sizes(self):
        tensor_numel = 40

        def _get_and_check_split_sizes(
            world_size,
            in_offsets,
            out_offsets,
            in_split_sizes,
        ):

            for my_rank in range(world_size):
                _in_split_sizes = in_split_sizes[my_rank]
                _out_split_sizes = [
                    in_split_sizes[i][my_rank] for i in range(world_size)
                ]
                res_in_split_sizes, res_out_split_sizes = _offsets_to_split_sizes(
                    in_offsets, out_offsets, tensor_numel, world_size, my_rank
                )
                self.assertEquals(_in_split_sizes, res_in_split_sizes)
                self.assertEquals(_out_split_sizes, res_out_split_sizes)

        # The tensor size can be evenly divided by the world size.
        world_size = 4
        in_offsets = [0, 10, 20, 30]
        out_offsets = [0, 10, 20, 30]
        in_split_sizes = [
            [10, 0, 0, 0],
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 10],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 4
        in_offsets = [0, 3, 17, 18]
        out_offsets = [0, 10, 20, 30]
        in_split_sizes = [
            [3, 0, 0, 0],
            [7, 7, 0, 0],
            [0, 1, 0, 0],
            [0, 2, 10, 10],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 4
        in_offsets = [0, 10, 20, 30]
        out_offsets = [0, 3, 17, 18]
        in_split_sizes = [
            [3, 7, 0, 0],
            [0, 7, 1, 2],
            [0, 0, 0, 10],
            [0, 0, 0, 10],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 4
        in_offsets = [0, 7, 11, 25]
        out_offsets = [0, 10, 17, 18]
        in_split_sizes = [
            [7, 0, 0, 0],
            [3, 1, 0, 0],
            [0, 6, 1, 7],
            [0, 0, 0, 15],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        # The tensor size cannot be evenly divided by the world size.
        world_size = 6
        in_offsets = [0, 7, 14, 21, 28, 35]
        out_offsets = [0, 7, 14, 21, 28, 35]
        in_split_sizes = [
            [7, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 0, 0],
            [0, 0, 7, 0, 0, 0],
            [0, 0, 0, 7, 0, 0],
            [0, 0, 0, 0, 7, 0],
            [0, 0, 0, 0, 0, 5],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

        world_size = 6
        in_offsets = [0, 0, 10, 11, 28, 40]
        out_offsets = [0, 7, 14, 21, 28, 35]
        in_split_sizes = [
            [0, 0, 0, 0, 0, 0],
            [7, 3, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 3, 7, 7, 0, 0],
            [0, 0, 0, 0, 7, 5],
            [0, 0, 0, 0, 0, 0],
        ]
        _get_and_check_split_sizes(world_size, in_offsets, out_offsets, in_split_sizes)

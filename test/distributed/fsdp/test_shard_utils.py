# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import _create_chunk_sharded_tensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestShardUtilsDistributed(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _create_tensor(self, *size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).cuda()

    @skip_if_lt_x_gpu(2)
    def test_create_chunk_sharded_tensor(self):
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)

            sharded_tensor = _create_chunk_sharded_tensor(
                tensor,
                self.rank,
                self.world_size,
                torch.cuda.device_count(),
                _get_default_group(),
            )
            output = torch.empty(*size).cuda() if self.rank == 0 else None
            sharded_tensor.gather(0, output)
            if self.rank == 0:
                self.assertEqual(tensor, output)


if __name__ == "__main__":
    run_tests()

import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests


class TestFullyShard1DTrainingCore(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group(self):
        torch.manual_seed(42)
        lin_dim = 32
        model = MLP(lin_dim, torch.device("cuda"))
        fully_shard(model.in_proj, reshard_after_forward=2)
        fully_shard(model, reshard_after_forward=2)
        model = torch.compile(model)
        inp = torch.randn((8, lin_dim), device=torch.device("cuda"))
        model(inp)

if __name__ == "__main__":
    run_tests()

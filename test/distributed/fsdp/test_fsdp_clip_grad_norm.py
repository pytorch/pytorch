# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, GELU
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
)
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, run_tests


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestClipGradNorm(FSDPTest):
    class SimpleNN(torch.nn.Module):
        def __init__(self, linear_size, rank=None):
            super().__init__()
            self.fc1 = Linear(*linear_size[0])
            self.gelu = GELU()
            self.fc2 = Linear(*linear_size[1])

        def forward(self, inp):
            return self.fc2(self.gelu(self.fc1(inp)))

    @skip_if_lt_x_gpu(2)
    def test_one_iteration(self):
        """Test FSDP with clip grad norm."""
        model = self.SimpleNN([[3, 15], [15, 17]])
        group = dist.distributed_c10d._get_default_group()
        input = torch.rand(8, 3)
        norm_type = 2.5
        model.to(self.rank)
        model = FSDP(model)
        self.assertTrue(len(input) >= self.world_size)
        in_data = torch.Tensor(input[self.rank]).to(self.rank)
        out = model(in_data)
        out.float().sum().backward()
        fsdp_modules_list = []
        total_norms = []
        for p in model.parameters():
            local_norm = torch.linalg.norm(p.grad, norm_type, dtype=torch.float32)
            total_norm = local_norm ** norm_type
            dist.all_reduce(total_norm, group=group)
            total_norms.append(total_norm ** (1.0 / norm_type))
        norm_cap = min(total_norms) / 2.0
        model._clip_grad_norm_(norm_cap, norm_type=norm_type)
        total_norms_after_clip = []
        for p in model.parameters():
            local_norm = torch.linalg.norm(p.grad, norm_type, dtype=torch.float32)
            total_norm = local_norm ** norm_type
            dist.all_reduce(total_norm, group=group)
            total_norms_after_clip.append(total_norm ** (1.0 / norm_type))
        self.assertTrue(all(norm <= norm_cap for norm in total_norms_after_clip))


if __name__ == "__main__":
    run_tests()

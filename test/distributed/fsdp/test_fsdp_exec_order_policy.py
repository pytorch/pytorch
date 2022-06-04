# Owner(s): ["oncall: distributed"]

import copy
import sys
import warnings
from contextlib import suppress

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    ShardingStrategy,
    _ExecOrderPolicy,
)
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class TestFSDPExecOrderPolicy(FSDPTest):
    def _test_parity_with_ddp(
        self,
        model_ctor,
        bucket_size: int,
        num_iters: int,
    ):
        torch.manual_seed(42)
        # TODO (awgu): Add *args, **kwargs as needed to `model_ctor()`
        model = model_ctor().cuda()
        group = dist.distributed_c10d._get_default_group()

        fsdp_model = FullyShardedDataParallel(
            copy.deepcopy(model), group, auto_wrap_policy=_ExecOrderPolicy(bucket_size),
        )
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)
        ddp_model = DistributedDataParallel(model, device_ids=[self.rank], process_group=group)
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)

        for _ in range(num_iters):
            inp = torch.randn((4, 3, 32, 32)).to(self.rank)
            iter_losses = []
            for model, optim in ((fsdp_model, fsdp_optim), (ddp_model, ddp_optim)):
                optim.zero_grad()
                out = model(inp)
                loss = out.sum()
                iter_losses.append(loss.item())
                loss.backward()
                optim.step()
            self.assertEqual(iter_losses[0], iter_losses[1])





    @skip_if_lt_x_gpu(2)
    def test_constructor(self):
        group = dist.distributed_c10d._get_default_group()
        model = self._get_nonwrapped_model(group).cuda()
        fsdp_model = FSDP(model, group, auto_wrap_policy=_ExecOrderPolicy(4e3))
        print(f"[Rank {self.rank}] printing parameters!")
        for param_name, param in fsdp_model.named_parameters():
            print(f"[Rank {self.rank}] {param_name} {param.shape}")

    @skip_if_lt_x_gpu(2)
    def test_fwd_bwd(self):
        self._test_parity_with_ddp(CNN, 1e2, 3)
        # group = dist.distributed_c10d._get_default_group()
        # model = self._get_nonwrapped_model(group).cuda()
        # fsdp_model = FSDP(model, group, auto_wrap_policy=_ExecOrderPolicy(4e3))
        # # model = torch.nn.Sequential(
        # #     torch.nn.Linear(2, 3),
        # #     torch.nn.ReLU(),
        # #     torch.nn.Sequential(torch.nn.Linear(3, 2)),
        # # )
        # # fsdp_model = FSDP(model.cuda(), group, auto_wrap_policy=_ExecOrderPolicy(1))
        # optim = torch.optim.Adam(list(fsdp_model.parameters()), lr=1e-3)
        # # module = fsdp_model.module
        # device = torch.device("cuda")
        # for _ in range(3):
        #     inp = fsdp_model.module.get_input(device)
        #     output = fsdp_model(*inp)
        #     # inp = (torch.randn((8, 2)).to(device),)
        #     # loss = module.get_loss(inp, output).to(device)
        #     # module.run_backward(loss)
        #     loss = output.sum()
        #     loss.backward()
        #     optim.step()


instantiate_parametrized_tests(TestFSDPExecOrderPolicy)

if __name__ == "__main__":
    run_tests()

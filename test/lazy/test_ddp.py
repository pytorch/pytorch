import copy
import os
import sys
import torch
import torch._lazy
import torch._lazy.ts_backend
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from datetime import timedelta
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN, NO_MULTIPROCESSING_SPAWN
from torch.testing._internal.common_distributed import MultiProcessTestCase

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print("Skip dev-asan as torch + multiprocessing spawn have known issues", file=sys.stderr)
    sys.exit(0)

if NO_MULTIPROCESSING_SPAWN:
    print("Spawn not available, skipping tests.", file=sys.stderr)
    sys.exit(0)

torch._lazy.ts_backend.init()
os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def step(model, input, labels, device, loss_fn, optimizer):
    input = input.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    output = model(input)

    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()
    # FIXME(alanwaketan): Investigate why we prodcue different
    # results if mark_step.
    # if device.type == "lazy":
    #     torch._lazy.mark_step(str(device))
    return loss

def init(model, device, rank, size):
    model = copy.deepcopy(model).to(device)
    if device.type == "lazy":
        # Somehow, LazyTensor crashes if gradient_as_bucket_view is not set.
        # FIXME(alanwaketan): Investigate why and if we can remove this constraint.
        model = DDP(model, gradient_as_bucket_view=True)
        # FIXME(alanwaketan): Do we need this or not?
        # model.register_comm_hook(None, torch._lazy.lazy_comm_hook)
    if device.type == "cuda":
        model = DDP(
            model,
            gradient_as_bucket_view=True,
            device_ids=[rank],
            process_group=dist.ProcessGroupNCCL(
                dist.distributed_c10d._get_default_store(), rank, size, timedelta(minutes=3))
        )

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    return model, loss_fn, optimizer

class TestDistributedLazy(MultiProcessTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(MASTER_PORT)
        os.environ["LTC_TS_CUDA"] = "1"

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        # initialize the process group
        dist.Backend.register_backend("lazy", torch._lazy.create_lazy_process_group)
        dist.init_process_group("lazy", rank=self.rank, world_size=self.world_size)

        self.run_test(test_name, pipe)
        dist.destroy_process_group()
        sys.exit(0)

    # Needed since MultiProcessTestCase assumes a world_size of 4, but we
    # run these tests under other various world_sizes.
    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"])

    def _all_close(self, parameters_a, parameters_b):
        for param_a, param_b in zip(parameters_a, parameters_b):
            self.assertEqual(param_a.cpu(), param_b.cpu())

    def test_DistributedDataParallelLazy(self):
        # disable all JIT optimizations and fusions.
        torch._C._jit_set_bailout_depth(0)

        model = ToyModel()

        device_lazy = torch.device("lazy", self.rank)
        model_lazy, loss_fn_lazy, optimizer_lazy = init(model, device_lazy, self.rank, self.world_size)

        device_cuda = torch.device("cuda", self.rank)
        model_cuda, loss_fn_cuda, optimizer_cuda = init(model, device_cuda, self.rank, self.world_size)

        self._all_close(model_lazy.parameters(), model_cuda.parameters())
        for i in range(5):
            input = torch.randn(20, 10)
            labels = torch.randn(20, 5)

            loss_lazy = step(model_lazy, input, labels, device_lazy, loss_fn_lazy, optimizer_lazy)
            loss_cuda = step(model_cuda, input, labels, device_cuda, loss_fn_cuda, optimizer_cuda)

            self._all_close([loss_lazy], [loss_cuda])
            self._all_close(model_lazy.parameters(), model_cuda.parameters())
            print(f"{os.getpid()}: iteration {i} lazy_parameters ~= cuda_parameters, loss_lazy={loss_lazy}, loss_cuda={loss_cuda}")


if __name__ == "__main__":
    run_tests()

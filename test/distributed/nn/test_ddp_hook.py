# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist
from torch import nn

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)

if TEST_WITH_DEV_DBG_ASAN:
    print("Multiprocessing spawn is not compatible with dev/dbg asan", file=sys.stderr)
    sys.exit(0)


class DummyTestModel(nn.Module):
    def __init__(self):
        super(DummyTestModel, self).__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


def relu_hook(module, input):
    return nn.functional.relu(input[0])


def gelu_hook(module, _input, output):
    return nn.functional.gelu(output)


def celu_hook(module, _input, output):
    return (nn.functional.celu(output[0]),)


class DistributedDataParallelHookTest(MultiProcessTestCase):
    def setUp(self):
        super(DistributedDataParallelHookTest, self).setUp()
        self._spawn_processes()

    def tearDown(self):
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_hook(self):
        local_model = DummyTestModel()
        ddp_model = DummyTestModel()
        store = dist.FileStore(self.file_name, self.world_size)
        process_group = dist.ProcessGroupNCCL(store, self.rank, self.world_size)
        local_model.fc.register_forward_pre_hook(relu_hook)
        local_model.fc.register_forward_hook(gelu_hook)
        ddp_model.fc.register_forward_pre_hook(relu_hook)
        ddp_model.fc.register_forward_hook(gelu_hook)
        local_model.fc.register_backward_hook(celu_hook)
        ddp_model.fc.register_backward_hook(celu_hook)
        ddp_model = DistributedDataParallel(
            ddp_model.to(self.rank), device_ids=[self.rank], process_group=process_group
        )
        store = dist.FileStore(self.file_name, self.world_size)
        input_data = torch.rand(5, 2)
        output_local = local_model(input_data)
        output_ddp = ddp_model(input_data.to(self.rank))
        self.assertEqual(output_local, output_ddp)
        output_local.sum().backward()
        output_ddp.sum().backward()
        ddp_grads = [p.grad for p in ddp_model.parameters()]
        self.assertEqual(ddp_grads[0], local_model.fc.weight.grad)
        self.assertEqual(ddp_grads[1], local_model.fc.bias.grad)


if __name__ == "__main__":
    run_tests()

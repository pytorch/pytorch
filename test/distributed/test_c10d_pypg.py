# Owner(s): ["oncall: distributed"]

import os

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    run_tests,
)
from torch.futures import Future
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import test_c10d_common
import weakref
from torch._C._distributed_c10d import _create_work_from_future
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
)

def create_work(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)

class MyWork(dist._Work):
    def __init__(self, result, pg):
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)
        self.pg_ = weakref.ref(pg)

    def wait(self, timeout):
        self.pg_().wait_count += 1
        return True

    def get_future(self):
        self.pg_().get_future_count += 1
        return self.future_

class LonelyRankProcessGroup(dist.ProcessGroup):
    """
    This PG only supports world_size of 1
    """
    def __init__(self, rank, world, use_wrapper):
        super().__init__(rank, world)
        assert rank == 0
        assert world == 1

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self.use_wrapper = use_wrapper
        self._work = []

    def broadcast(self, tensor_list, opts):
        if self.use_wrapper:
            return create_work(tensor_list)
        res = MyWork(tensor_list, self)
        self._work.append(res)
        return res

    def allgather(self, output_tensors, input_tensor, opts):
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)
        if self.use_wrapper:
            return create_work(output_tensors)

        res = MyWork(output_tensors, self)
        self._work.append(res)

        return res

    def allreduce(self, tensors, opts):
        if self.use_wrapper:
            return create_work(tensors)
        res = MyWork(tensors, self)
        self._work.append(res)
        return res

    def size(self):
        return self._world

    def getBackendName(self):
        return "lonely-pg"

    def __repr__(self):
        return f"PLG w:{self._world} r:{self._rank}"

# We cannot use parametrize as some tests are defined on the base class and use _get_process_group
class AbstractDDPSingleRank(test_c10d_common.CommonDistributedDataParallelTest):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self):
        return 1

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _get_process_group(self):
        return LonelyRankProcessGroup(self.rank, self.world_size, self.use_wrapper)

    def test_ddp_invoke_work_object(self):
        pg = self._get_process_group()

        torch.manual_seed(123)
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU()
        )
        wrapped_model = model
        input_tensor = torch.rand(2)
        model = DDP(model, process_group=pg)
        model(input_tensor).sum().backward()

        ddp_grad = wrapped_model[0].bias.grad.clone()

        wrapped_model.zero_grad()
        wrapped_model(input_tensor).sum().backward()
        self.assertEqual(wrapped_model[0].bias.grad, ddp_grad)
        if not self.use_wrapper:
            self.assertTrue(pg.wait_count > 0)
            self.assertTrue(pg.get_future_count > 0)

    def test_ddp_with_pypg(self):
        pg = self._get_process_group()

        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None)

    def test_ddp_with_pypg_with_grad_views(self):
        pg = self._get_process_group()

        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None, gradient_as_bucket_view=True)

class TestDDPWithWorkSubclass(AbstractDDPSingleRank, MultiProcessTestCase):
    @property
    def use_wrapper(self):
        return False

class TestDDPWithWorkWrapper(AbstractDDPSingleRank, MultiProcessTestCase):
    @property
    def use_wrapper(self):
        return True

if __name__ == '__main__':
    run_tests()

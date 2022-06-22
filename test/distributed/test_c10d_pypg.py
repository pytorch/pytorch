# Owner(s): ["oncall: distributed"]

import os
import sys
from functools import wraps, partial

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.futures import Future
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import test_c10d_common
import weakref
import tempfile


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
    def __init__(self, rank, world):
        super(LonelyRankProcessGroup, self).__init__(rank, world)
        assert rank == 0
        assert world == 1

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self._work = []

    def broadcast(self, tensor_list, opts):
        res = MyWork(tensor_list, self)
        self._work.append(res)
        return res

    def allgather(self, output_tensors, input_tensor, opts):
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)

        res = MyWork(output_tensors, self)
        self._work.append(res)

        return res

    def allreduce(self, tensors, opts):
        res = MyWork(tensors, self)
        self._work.append(res)
        return res

    def size(self):
        return self._world

    def getBackendName(self):
        return "lonely-pg"

    def __repr__(self):
        return f"PLG w:{self._world} r:{self._rank}"

class TestDDPSingleRank(test_c10d_common.CommonDistributedDataParallelTest, TestCase):
    def setUp(self):
        super(TestDDPSingleRank, self).setUp()
        # replicate what MultiProcessTest _spawn_proccess does
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        self.rank = 0

    @property
    def world_size(self):
        return 1

    def tearDown(self):
        super(TestDDPSingleRank, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _get_process_group(self):
        store = self._get_store()
        return LonelyRankProcessGroup(self.rank, self.world_size)

    def test_ddp_invoke_work_object(self):
        pg = LonelyRankProcessGroup(0, 1)      

        torch.manual_seed(123)
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU()
        )
        wrapped_model = model
        model = DDP(model, process_group=pg)
        model(torch.tensor([0.99, 0.77])).sum().backward()
        
        ddp_grad = wrapped_model[0].bias.grad.clone()

        wrapped_model.zero_grad()
        wrapped_model(torch.tensor([0.99, 0.77])).sum().backward()
        self.assertEqual(wrapped_model[0].bias.grad, ddp_grad)
        self.assertTrue(pg.wait_count > 0)
        self.assertTrue(pg.get_future_count > 0)

    def test_ddp_with_pypg(self):
        pg = LonelyRankProcessGroup(0, 1)      

        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None)

    def test_ddp_with_pypg_with_grad_views(self):
        pg = LonelyRankProcessGroup(0, 1)      

        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None, gradient_as_bucket_view=True)


if __name__ == '__main__':
    run_tests()

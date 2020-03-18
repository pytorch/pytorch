import unittest

import torch
import torch.distributed.autograd as dist_autograd
from torch.testing import FileCheck
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.distributed.rpc import rpc_async

@torch.jit.script
def local_add(t1, t2):
    return torch.add(t1, t2)

@torch.jit.script
def remote_add(t1, t2, dst: str):  # noqa: E999
    return rpc_async(dst, local_add, (t1, t2)).wait()

@torch.jit.script
def fork_add(t1, t2, dst: str):
    fut = torch.jit._fork(remote_add, t1, t2, dst)
    return torch.jit._wait(fut)

@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed autograd package does not support python2"
)
class JitDistAutogradTest(RpcAgentTestFixture):

    @dist_init
    def test_get_gradients(self):
        dst_rank = self.rank

        @torch.jit.script
        def dist_get_gradients(context_id):
            # type: (int) -> (Dict[Tensor, Tensor])
            return dist_autograd.get_gradients(context_id)

        FileCheck().check("get_gradients").run(str(dist_get_gradients.graph))
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = torch.add(t1, t2)

            dist_autograd.backward(context_id, [t3.sum()])
            grads = dist_get_gradients(context_id)

            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])

    @dist_init
    def test_jit_fork_within_context(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)
            res = fork_add(t1, t2, dst_worker_name)
            loss = res.sum()
            dist_autograd.backward(context_id, [loss])

            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)

# mypy: ignore-errors

from typing import Dict, Tuple

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import rpc_async
from torch.testing import FileCheck
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


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


class JitDistAutogradTest(RpcAgentTestFixture):
    @dist_init
    def test_get_gradients(self):
        dst_rank = self.rank

        @torch.jit.script
        def dist_get_gradients(context_id: int) -> (Dict[Tensor, Tensor]):
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
    def test_dist_backward(self):
        if self.rank != 0:
            return

        @torch.jit.script
        def dist_backward_script(context_id: int, loss: torch.Tensor):
            dist_autograd.backward(context_id, [loss])

        FileCheck().check("dist_backward").run(str(dist_backward_script.graph))
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(3, 3, requires_grad=True)
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)
            loss = rpc.rpc_sync(dst_worker_name, torch.add, args=(t1, t2)).sum()
            dist_backward_script(context_id, loss)

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

    @dist_init
    def test_restore_context_after_swtich_to_jit_thread(self):
        if self.rank != 0:
            return

        @torch.jit.script
        def forward_script(
            context_id: int, dst_worker_name: str, t1: Tensor, t2: Tensor
        ) -> Tuple[Tensor, Tensor]:
            res1_fut = rpc.rpc_async(dst_worker_name, local_add, (t1, t1))
            res1 = res1_fut.wait()  # After this, the script runs in a new JIT thread.
            loss1 = res1.sum()

            # SendRpcBackward is not attached, since DistAutogradContext is lost here.
            res2_fut = rpc.rpc_async(dst_worker_name, local_add, (t2, t2))
            res2 = res2_fut.wait()
            loss2 = res2.sum()

            return loss1, loss2

        with dist_autograd.context() as context_id:
            t1 = torch.ones((2, 3), requires_grad=True)
            t2 = torch.ones((2, 3), requires_grad=True)
            dst_worker_name = worker_name((self.rank + 1) % self.world_size)
            loss0, loss1 = forward_script(context_id, dst_worker_name, t1, t2)
            dist_autograd.backward(context_id, [loss0, loss1])
            grad0, grad1 = dist_autograd.get_gradients(context_id)
            self.assertEqual(grad0, grad1)

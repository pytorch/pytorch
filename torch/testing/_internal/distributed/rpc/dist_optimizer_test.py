from __future__ import absolute_import, division, print_function, unicode_literals

import threading
import unittest

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import optim
from torch.distributed.optim import DistributedOptimizer
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


class MyModule:
    lock = threading.Lock()

    def __init__(self):
        # avoid race where 2 modules could be initialized
        # concurrently thus changing the order random numbers are drawn.
        with MyModule.lock:
            torch.manual_seed(0)
            self.w = torch.rand((3, 3), requires_grad=True)

    def forward(self, t1):
        return torch.mm(self.w, t1)

    def get_w(self):
        return self.w


class FailingOptimizer(optim.Optimizer):
    def __init__(self, params):
        super(FailingOptimizer, self).__init__(params, {})

    def step(self, closure=None):
        raise ValueError("Error running optimizer.")


class OptimizerFailingOnConstructor(optim.Optimizer):
    def __init__(self, params):
        super(OptimizerFailingOnConstructor, self).__init__(params, {})
        raise ValueError("Error creating optimizer.")

    def step(self, closure=None):
        raise NotImplementedError


def _call_method(method, obj_rref, *args, **kwargs):
    return method(obj_rref.local_value(), *args, **kwargs)


def remote_method(method, obj_rref, *args, **kwargs):
    """
    Call rpc.remote on a method in a remote object.

    Args:
        method: the method (for example, Class.method)
        obj_rref (RRef): remote reference to the object
        args: positional arguments to pass to the method
        kwargs: keyword arguments to pass to the method

    Returns a RRef to the remote method call result.
    """
    return rpc.remote(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs,
    )


def rpc_async_method(method, obj_rref, *args, **kwargs):
    """
    Call rpc.rpc_async on a method in a remote object.

    Args:
        method: the method (for example, Class.method)
        obj_rref (RRef): remote reference to the object
        args: positional arguments to pass to the method
        kwargs: keyword arguments to pass to the method

    Returns a Future to the method call result.
    """
    return rpc.rpc_async(
        obj_rref.owner(),
        _call_method,
        args=[method, obj_rref] + list(args),
        kwargs=kwargs,
    )


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed optim does not support python2"
)
class DistOptimizerTest(RpcAgentTestFixture):
    @dist_init()
    def test_dist_optim_exception(self):
        # distributed version
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)

        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule)
        remote_param1 = remote_method(MyModule.get_w, remote_module1)
        remote_param2 = remote_method(MyModule.get_w, remote_module2)

        dist_optim = DistributedOptimizer(
            FailingOptimizer, [remote_param1, remote_param2]
        )

        with dist_autograd.context() as context_id:
            torch.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            output1 = rpc_async_method(MyModule.forward, remote_module1, t2)
            output2 = rpc_async_method(MyModule.forward, remote_module2, output1.wait())
            loss = torch.add(output2.wait(), t1).sum()

            dist_autograd.backward(context_id, [loss])
            with self.assertRaisesRegex(Exception, "Error running optimizer"):
                dist_optim.step(context_id)

    @dist_init()
    def test_dist_optim_exception_on_constructor(self):
        # distributed version
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)

        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule)
        remote_param1 = remote_method(MyModule.get_w, remote_module1)
        remote_param2 = remote_method(MyModule.get_w, remote_module2)

        with self.assertRaisesRegex(Exception, "Error creating optimizer."):
            dist_optim = DistributedOptimizer(
                OptimizerFailingOnConstructor, [remote_param1, remote_param2]
            )

    @dist_init()
    def test_dist_optim(self):
        # local version
        module1 = MyModule()
        module2 = MyModule()
        params = [module1.get_w(), module2.get_w()]
        local_optim = optim.SGD(params, lr=0.05)

        old_w1 = module1.w.clone().detach()
        old_w2 = module2.w.clone().detach()

        torch.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        output1 = module1.forward(t2)
        output2 = module2.forward(output1)
        loss = torch.add(output2, t1).sum()

        loss.backward()
        local_optim.step()

        # distributed version
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)

        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule)
        remote_param1 = remote_method(MyModule.get_w, remote_module1)
        remote_param2 = remote_method(MyModule.get_w, remote_module2)

        old_w1_remote = remote_param1.to_here()

        # sanity check: local and remote initial weights should match
        self.assertEqual(old_w1, remote_param1.to_here())
        self.assertEqual(old_w2, remote_param2.to_here())

        dist_optim = DistributedOptimizer(
            optim.SGD, [remote_param1, remote_param2], lr=0.05
        )

        with dist_autograd.context() as context_id:
            torch.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            output1 = rpc_async_method(MyModule.forward, remote_module1, t2)
            output2 = rpc_async_method(MyModule.forward, remote_module2, output1.wait())
            loss = torch.add(output2.wait(), t1)

            dist_autograd.backward(context_id, [loss.sum()])
            dist_optim.step(context_id)

            new_w1 = rpc_async_method(MyModule.get_w, remote_module1).wait()
            new_w2 = rpc_async_method(MyModule.get_w, remote_module2).wait()

            # ensure optimizer changed weights
            self.assertNotEqual(old_w1, new_w1)
            self.assertNotEqual(old_w2, new_w2)
            # ensure local equals remote
            self.assertEqual(new_w1, module1.get_w())
            self.assertEqual(new_w2, module2.get_w())

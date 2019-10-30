from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

from dist_utils import INIT_METHOD_TEMPLATE, dist_init
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.optimizer.dist_optimizer as dist_optimizer
import torch.distributed.rpc as rpc

class MyModule:
    def __init__(self):
        torch.manual_seed(0)
        self.w = torch.rand((3, 3), requires_grad=True)

    def forward(self, t1):
        return torch.mm(self.w, t1)

    def get_w(self):
        return self.w


class FailingOptimizer(dist_optimizer.FunctionalOptimizer):
    def step(self, gradients):
        raise ValueError('Error running optimizer.')


def _call_meth(meth, obj_rref, *args, **kwargs):
    return meth(obj_rref.local_value().wait(), *args, **kwargs)


def remote_meth(meth, obj_rref, *args, **kwargs):
    """
    Call rpc.remote on a method in a remote object.

    Args:
        meth: the method (for example, Class.method)
        obj_rref (RRef): remote reference to the object
        args: positional arguments to pass to the method
        kwargs: keyword arguments to pass to the method

    Returns a RRef to the remote method call result.
    """
    return rpc.remote(
        obj_rref.owner(),
        _call_meth,
        args=[meth, obj_rref] + list(args),
        kwargs=kwargs
    )


def rpc_async_meth(meth, obj_rref, *args, **kwargs):
    """
    Call rpc.remote on a method in a remote object.

    Args:
        meth: the method (for example, Class.method)
        obj_rref (RRef): remote reference to the object
        args: positional arguments to pass to the method
        kwargs: keyword arguments to pass to the method

    Returns a Future to the method call result.
    """
    return rpc.rpc_async(
        obj_rref.owner(),
        _call_meth,
        args=[meth, obj_rref] + list(args),
        kwargs=kwargs
    )


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed autograd package " "does not support python2"
)
class DistOptimizerTest(object):

    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return INIT_METHOD_TEMPLATE.format(
            file_name=self.file_name, rank=self.rank, world_size=self.world_size
        )

    @dist_init()
    def test_dist_optim_exception(self):
        # distributed version
        owner1 = 'worker%d' % ((self.rank + 1) % self.world_size)
        owner2 = 'worker%d' % ((self.rank + 2) % self.world_size)

        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule)
        remote_param1 = remote_meth(MyModule.get_w, remote_module1)
        remote_param2 = remote_meth(MyModule.get_w, remote_module2)

        dst_optim = dist_optimizer.DistributedOptimizer(
            FailingOptimizer,
            [remote_param1, remote_param2],
        )

        with dist_autograd.context() as context_id:
            torch.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            output1 = rpc_async_meth(MyModule.forward, remote_module1, t2)
            output2 = rpc_async_meth(
                MyModule.forward, remote_module2, output1.wait())
            loss = torch.add(output2.wait(), t1)

            dist_autograd.backward([loss.sum()])
            with self.assertRaisesRegex(Exception, "Error running optimizer"):
                dst_optim.step(context_id)

    @dist_init()
    def test_dist_optim(self):
        if self.rank != 0:
            return

        # local version
        module1 = MyModule()
        module2 = MyModule()
        params = [module1.get_w(), module2.get_w()]
        optim = dist_optimizer.FunctionalSGD(params, lr=0.05)

        old_w1 = module1.w.clone().detach()
        old_w2 = module2.w.clone().detach()

        torch.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        output1 = module1.forward(t2)
        output2 = module2.forward(output1)
        loss = torch.add(output2, t1)

        grads = torch.autograd.grad([loss.sum()], params + [t1, t2])
        param_grads = grads[:len(params)]
        optim.step(param_grads)

        # distributed version
        owner1 = 'worker%d' % ((self.rank + 1) % self.world_size)
        owner2 = 'worker%d' % ((self.rank + 2) % self.world_size)

        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule)
        remote_param1 = remote_meth(MyModule.get_w, remote_module1)
        remote_param2 = remote_meth(MyModule.get_w, remote_module2)

        dst_optim = dist_optimizer.DistributedOptimizer(
            dist_optimizer.FunctionalSGD,
            [remote_param1, remote_param2],
            lr=0.05,
        )

        with dist_autograd.context() as context_id:
            torch.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            output1 = rpc_async_meth(MyModule.forward, remote_module1, t2)
            output2 = rpc_async_meth(
                MyModule.forward, remote_module2, output1.wait())
            loss = torch.add(output2.wait(), t1)

            dist_autograd.backward([loss.sum()])
            dst_optim.step(context_id)

            new_w1 = rpc_async_meth(MyModule.get_w, remote_module1).wait()
            new_w2 = rpc_async_meth(MyModule.get_w, remote_module2).wait()

            # ensure optimizer changed weights
            self.assertNotEqual(old_w1, new_w1)
            self.assertNotEqual(old_w2, new_w2)
            # ensure local equals remote
            self.assertEqual(new_w1, module1.get_w())
            self.assertEqual(new_w2, module2.get_w())


if __name__ == '__main__':
    unittest.main()

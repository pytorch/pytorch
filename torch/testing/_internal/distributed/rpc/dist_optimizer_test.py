
import threading

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

    def __init__(self, requires_grad=True):
        # cannot directly use torch.manual_seed(0) as all threads share the same
        # default generator. The race from multiple RPC threads could mess up
        # the draw order from the default RNG instance, leading to
        # non-deterministic behavior. Hence, create a dedicated RNG here.
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        self.w = torch.rand((3, 3), requires_grad=requires_grad, generator=g_cpu)

    def forward(self, t1):
        return torch.mm(self.w, t1)

    def get_w(self):
        return self.w


class FailingOptimizer(optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

    def step(self, closure=None):
        raise ValueError("Error running optimizer.")


class OptimizerFailingOnConstructor(optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})
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
            g_cpu = torch.Generator()
            g_cpu.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
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

    def _test_dist_optim_base(self, optim_cls, *args, **kwargs):
        # local version
        module1 = MyModule()
        module2 = MyModule()
        params = [module1.get_w(), module2.get_w()]
        local_optim = optim_cls(params, *args, **kwargs)

        old_w1 = module1.w.clone().detach()
        old_w2 = module2.w.clone().detach()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
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
            optim_cls, [remote_param1, remote_param2], *args, **kwargs
        )

        with dist_autograd.context() as context_id:
            g_cpu.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
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

    @dist_init()
    def test_dist_optim(self):
        self._test_dist_optim_base(optim.Adagrad, lr=0.05)
        self._test_dist_optim_base(optim.Adam, lr=1e-2, amsgrad=True)
        self._test_dist_optim_base(optim.AdamW, lr=0.05, amsgrad=True)
        self._test_dist_optim_base(optim.SGD, lr=0.05)
        self._test_dist_optim_base(optim.SGD, lr=1e-3, momentum=1, weight_decay=1, nesterov=True)
        self._test_dist_optim_base(optim.Adadelta, rho=0.95)
        self._test_dist_optim_base(optim.RMSprop, lr=0.05)
        self._test_dist_optim_base(optim.Adamax, lr=0.05)
        self._test_dist_optim_base(optim.Rprop, lr=0.05)

    def _test_dist_optim_none_grads(self, optim_cls, *args, **kwargs):
        # local version
        module1 = MyModule()
        module2 = MyModule(requires_grad=False)
        params = [module1.get_w(), module2.get_w()]
        local_optim = optim_cls(params, *args, **kwargs)

        old_w1 = module1.w.clone().detach()
        old_w2 = module2.w.clone().detach()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        output1 = module1.forward(t2)
        output2 = module2.forward(output1)
        loss = torch.add(output2, t1).sum()

        loss.backward()
        local_optim.step()

        # distributed version
        owner1 = "worker%d" % ((self.rank + 1) % self.world_size)
        owner2 = "worker%d" % ((self.rank + 2) % self.world_size)

        remote_module1 = rpc.remote(owner1, MyModule)
        remote_module2 = rpc.remote(owner2, MyModule, args=(False,))
        remote_param1 = remote_module1.remote().get_w()
        remote_param2 = remote_module2.remote().get_w()

        # sanity check: local and remote initial weights should match
        self.assertEqual(old_w1, remote_param1.to_here())
        self.assertEqual(old_w2, remote_param2.to_here())

        dist_optim = DistributedOptimizer(
            optim_cls, [remote_param1, remote_param2], *args, **kwargs
        )

        with dist_autograd.context() as context_id:
            g_cpu.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            output1 = remote_module1.rpc_async().forward(t2)
            output2 = remote_module2.rpc_async().forward(output1.wait())
            loss = torch.add(output2.wait(), t1)

            dist_autograd.backward(context_id, [loss.sum()])
            dist_optim.step(context_id)

            new_w1 = remote_module1.rpc_async().get_w().wait()
            new_w2 = remote_module2.rpc_async().get_w().wait()

            # ensure optimizer changed weights for w1
            self.assertNotEqual(old_w1, new_w1)

            # ensure optimizer not changed weights for w2
            self.assertEqual(old_w2, new_w2)
            # ensure local equals remote
            self.assertEqual(new_w1, module1.get_w())
            self.assertEqual(new_w2, module2.get_w())

    @dist_init()
    def test_dist_optim_none_grads(self):
        self._test_dist_optim_none_grads(optim.SGD, lr=0.05)
        self._test_dist_optim_none_grads(optim.RMSprop, lr=0.05)
        self._test_dist_optim_none_grads(optim.Rprop, lr=0.05)
        self._test_dist_optim_none_grads(optim.Adadelta, rho=0.95)

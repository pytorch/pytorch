import unittest

import torch
import torch.distributed.rpc as rpc
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def rpc_return_rref(dst):
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))


@torch.jit.script
def one_arg(value):
    return value + 1


@torch.jit.script
class MyScriptClass:
    def __init__(self):
        self.a = 10


@torch.jit.interface
class MyModuleInterface(torch.nn.Module):
    def forward(self):
        # type: () -> Tensor
        pass


class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.a = torch.randn(10)

    @torch.jit.script_method
    def forward(self):
        # type: () -> Tensor
        return self.a


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed rpc package does not support python2"
)
class JitRpcTest(RpcAgentTestFixture):
    @dist_init
    def test_torchscript_function(self):
        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)
        local_ret = one_arg(torch.ones(2, 2))
        ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        self.assertEqual(ret, local_ret)
        rref = rpc.remote(dst_worker_name, one_arg, args=(torch.ones(2, 2),))
        self.assertEqual(rref.to_here(), local_ret)
        # create rref to itself
        local_rref = rpc.remote(
            "worker{}".format(self.rank), one_arg, args=(torch.ones(2, 2),)
        )
        self.assertEqual(local_rref.to_here(), local_ret)

    # @dist_init
    # def test_torchscript_function_exception(self):
    #     dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)
    #     with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
    #         ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(10, 20))
    #
    #     with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
    #         rref = rpc.remote(dst_worker_name, one_arg, args=(10, 20))
    #
    # @dist_init
    # def test_torchscript_functions_not_supported(self):
    #     # Right now _rpc_sync_torchscript does not accept annotated torchscript
    #     # class name or script module class name or their class method names.
    #     # But rpc_sync still accepts script class name and run it in
    #     # the same code path as python call.
    #     # Currently neither rpc_sync or _rpc_sync_torchscript is allowed to
    #     # accept script module and script module method.
    #     n = self.rank + 1
    #     dst_rank = n % self.world_size
    #     with self.assertRaisesRegex(
    #         RuntimeError, "attempted to get undefined function"
    #     ):
    #         ret = rpc._rpc_sync_torchscript(
    #             "worker{}".format(dst_rank),
    #             torch.jit._qualified_name(MyScriptClass),
    #             args=(),
    #         )
    #     ret = rpc.rpc_sync("worker{}".format(dst_rank), MyScriptClass, args=())
    #
    #     with self.assertRaisesRegex(
    #         RuntimeError, "attempted to get undefined function"
    #     ):
    #         ret = rpc._rpc_sync_torchscript(
    #             "worker{}".format(dst_rank),
    #             torch.jit._qualified_name(MyScriptModule),
    #             args=(),
    #         )
    #
    #     with self.assertRaisesRegex(
    #         RuntimeError, "attempted to get undefined function"
    #     ):
    #         ret = rpc._rpc_sync_torchscript(
    #             "worker{}".format(dst_rank),
    #             torch.jit._qualified_name(MyScriptModule().forward),
    #             args=(),
    #         )
    #     # Python 3.5 and Python 3.6 throw different error message, the only
    #     # common word can be greped is "pickle".
    #     with self.assertRaisesRegex(Exception, "pickle"):
    #         ret = rpc.rpc_sync(
    #             "worker{}".format(dst_rank), MyScriptModule().forward, args=()
    #         )
    #
    # @dist_init
    # def test_rref_as_arg(self):
    #     n = self.rank + 1
    #     dst_rank = n % self.world_size
    #     rref_var = rpc_return_rref("worker{}".format(dst_rank))
    #
    #     @torch.jit.script
    #     def rref_tensor_to_here(rref_var):
    #         # type: (RRef[Tensor]) -> Tensor
    #         return rref_var.to_here()
    #
    #     res = rref_tensor_to_here(rref_var)
    #     self.assertEqual(res, torch.ones(2, 2) + 1)
    #
    # @dist_init
    # def test_rref_is_owner(self):
    #     n = self.rank + 1
    #     dst_rank = n % self.world_size
    #     rref_var = rpc_return_rref("worker{}".format(dst_rank))
    #
    #     @torch.jit.script
    #     def rref_tensor_is_owner(rref_var):
    #         # type: (RRef[Tensor]) -> bool
    #         return rref_var.is_owner()
    #
    #     res = rref_tensor_is_owner(rref_var)
    #     self.assertEqual(res, False)
    #
    # @dist_init
    # def test_remote_script_module(self):
    #     @torch.jit.ignore
    #     def my_script_module_init():
    #         # type: () -> MyModuleInterface
    #         return MyScriptModule()
    #
    #     @torch.jit.script
    #     def construct_my_script_module():
    #         # type: () -> MyModuleInterface
    #         return my_script_module_init()
    #
    #     dst_worker_name = "worker{}".format(self.rank % self.world_size)
    #     ref_script_module = rpc.remote(
    #         dst_worker_name, construct_my_script_module, args=()
    #     )
    #
    #     @torch.jit.script
    #     def owner_run_ref_script_module(ref_script_module):
    #         # type: (RRef[MyModuleInterface]) -> Tensor
    #         module = ref_script_module.to_here()
    #         return module.forward()
    #
    #     owner_local_ret = owner_run_ref_script_module(ref_script_module)
    #     print(owner_local_ret, type(owner_local_ret))
    #     """
    #     ret = rpc.rpc_sync(
    #         "worker{}".format(dst_rank),
    #         run_ref_script_module,
    #         args=(ref_script_module,))
    #     self.assertEqual(ret, local_ret)
    #     """

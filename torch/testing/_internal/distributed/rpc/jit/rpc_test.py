import unittest
from typing import Dict, Tuple

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def python_function():
    return 0


def rpc_return_rref(dst):
    return rpc.remote(dst, torch.add, args=(torch.ones(2, 2), 1))


@torch.jit.script
def no_arg():
    return 0


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
    def __init__(self, rank):
        super().__init__()
        self.a = torch.ones(rank)

    @torch.jit.script_method
    def forward(self):
        # type: () -> Tensor
        return self.a


# Define Script functions on both client and server sides.
@torch.jit.script
def two_args_two_kwargs(
    first_arg,
    second_arg,
    first_kwarg=torch.tensor([3, 3]),
    second_kwarg=torch.tensor([4, 4]),
):
    return first_arg + second_arg + first_kwarg + second_kwarg


@torch.jit.script
def assorted_types_args_kwargs(
    tensor_arg: Tensor,  # noqa: E999
    str_arg: str,
    int_arg: int,
    tensor_kwarg: Tensor = torch.tensor([2, 2]),
    str_kwarg: str = "str_kwarg",
    int_kwarg: int = 2,
):
    return tensor_arg + tensor_kwarg, str_arg + str_kwarg, int_arg + int_kwarg


@torch.jit.script
def raise_script():
    raise RuntimeError("Expected error")
    return 0


@torch.jit.script
def rpc_async_call_remote_torchscript_in_torchscript(
    dst_worker_name: str, args: Tuple[Tensor, Tensor], kwargs: Dict[str, Tensor]
):
    fut = rpc.api.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
    ret = fut.wait()
    return ret


class JitRpcAsyncOpTest:
    # Call functions remotely from Script.
    @dist_init
    def test_all_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {}
        ret = rpc_async_call_remote_torchscript_in_torchscript(
            dst_worker_name, args, kwargs
        )
        self.assertEqual(ret, torch.tensor([10, 10]))

    @dist_init
    def test_some_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {"first_kwarg": torch.tensor([2, 2])}
        ret = rpc_async_call_remote_torchscript_in_torchscript(
            dst_worker_name, args, kwargs
        )
        self.assertEqual(ret, torch.tensor([9, 9]))

    @dist_init
    def test_no_kwargs_are_populated_by_defaults(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        ret = rpc_async_call_remote_torchscript_in_torchscript(
            dst_worker_name, args, kwargs
        )
        self.assertEqual(ret, torch.tensor([8, 8]))

    @dist_init
    def test_kwargs_in_the_front_can_be_specified_by_extra_args(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_with_extra_arg(
            dst_worker_name: str,  # noqa: E999
        ):
            args = (
                torch.tensor([1, 1]),
                torch.tensor([2, 2]),
                # This extra arg will be fed to the first kwarg.
                torch.tensor([2, 2]),
            )
            kwargs = {"second_kwarg": torch.tensor([3, 3])}
            fut = rpc.api.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            ret = fut.wait()
            return ret

        ret = rpc_async_call_remote_torchscript_in_torchscript_with_extra_arg(
            dst_worker_name
        )
        self.assertEqual(ret, torch.tensor([8, 8]))

    @dist_init
    def test_args_and_kwargs_contain_different_types(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_with_assorted_types(
            dst_worker_name: str
        ):
            args = (torch.tensor([1, 1]), "str_arg", 1)
            # Must annotate the value type as `Any`, because JIT type inference
            # does not support multiple types when defining a Dict.
            # The error JIT gives is,
            # "Dict values must contain only a single type, "
            # "expected: Tensor but found str instead."
            kwargs: Dict[str, Any] = {
                "tensor_kwarg": torch.tensor([3, 3]),
                "str_kwarg": "_str_kwarg",
                "int_kwarg": 3,
            }
            fut = rpc.api.rpc_async(
                dst_worker_name, assorted_types_args_kwargs, args, kwargs
            )
            ret = fut.wait()
            return ret

        ret = rpc_async_call_remote_torchscript_in_torchscript_with_assorted_types(
            dst_worker_name
        )
        self.assertEqual(ret, (torch.tensor([4, 4]), "str_arg_str_kwarg", 4))

    @dist_init
    def test_kwargs_not_passed(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_without_kwargs_passed(
            dst_worker_name: str
        ):
            args = ()
            fut = rpc.api.rpc_async(dst_worker_name, no_arg, args)
            ret = fut.wait()
            return ret

        ret = rpc_async_call_remote_torchscript_in_torchscript_without_kwargs_passed(
            dst_worker_name
        )
        self.assertEqual(ret, 0)

    @dist_init
    def test_args_kwargs_are_neither_passed(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_without_args_kwargs_passed(
            dst_worker_name: str
        ):
            fut = rpc.api.rpc_async(dst_worker_name, no_arg)
            ret = fut.wait()
            return ret

        ret = rpc_async_call_remote_torchscript_in_torchscript_without_args_kwargs_passed(
            dst_worker_name
        )
        self.assertEqual(ret, 0)

    @dist_init
    def test_less_than_needed_args_are_specified(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        # Notice, args matching happens during scripting.
        with self.assertRaisesRegex(RuntimeError, "Argument second_arg not provided"):

            @torch.jit.script
            def rpc_async_call_remote_torchscript_in_torchscript_with_less_args(
                dst_worker_name: str,  # noqa: E999
            ):
                args = (torch.tensor([1, 1]),)
                kwargs = {}
                fut = rpc.api.rpc_async(
                    dst_worker_name, two_args_two_kwargs, args, kwargs
                )
                ret = fut.wait()
                return ret

    @dist_init
    def test_more_than_needed_args_are_specified(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        # Notice, args matching happens during scripting.
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected at most 4 arguments but found 5 positional arguments",
        ):

            @torch.jit.script
            def rpc_async_call_remote_torchscript_in_torchscript_with_more_args(
                dst_worker_name: str,
            ):
                args = (
                    torch.tensor([1, 1]),
                    torch.tensor([2, 2]),
                    torch.tensor([3, 3]),
                    torch.tensor([4, 4]),
                    torch.tensor([5, 5]),
                )
                kwargs = {}
                fut = rpc.api.rpc_async(
                    dst_worker_name, two_args_two_kwargs, args, kwargs
                )
                ret = fut.wait()
                return ret

    @dist_init
    def test_unexepected_kwarg_is_specified(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        # Notice, kwargs matching happens during execution.
        @torch.jit.script
        def rpc_async_call_remote_torchscript_in_torchscript_with_unexpected_kwarg(
            dst_worker_name: str,  # noqa: E999
        ):
            args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
            kwargs = {"third_kwarg": torch.tensor([1, 1])}
            fut = rpc.api.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(
            RuntimeError, "Unknown keyword argument 'third_kwarg'"
        ):
            ret = rpc_async_call_remote_torchscript_in_torchscript_with_unexpected_kwarg(
                dst_worker_name
            )
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_python_function_remotely_from_script_not_supported(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        @torch.jit.script
        def rpc_async_call_remote_py_function_in_torchscript(dst_worker_name: str):
            args = ()
            kwargs = {}
            fut = rpc.api.rpc_async(dst_worker_name, python_function, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function"
        ):
            ret = rpc_async_call_remote_py_function_in_torchscript(dst_worker_name)
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_raises_remotely_from_script(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        # Notice, TorchScript always translates(emits) Python `raise` statement,
        # as the exception message string, "Exception",
        # no matter what exception type and excetpion message are in the statement,
        @torch.jit.script
        def rpc_async_call_remote_raising_torchscript_in_torchscript(
            dst_worker_name: str
        ):
            args = ()
            kwargs = {}
            fut = rpc.api.rpc_async(dst_worker_name, raise_script, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(RuntimeError, "Exception"):
            ret = rpc_async_call_remote_raising_torchscript_in_torchscript(
                dst_worker_name
            )
            self.assertEqual(ret, 0)

    @dist_init
    def test_call_script_function_that_not_exists_remotely_from_script(self):
        if self.rank != 0:
            return

        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)

        @torch.jit.script
        def nonexisting_script():
            return 0

        @torch.jit.script
        def rpc_async_call_remote_nonexisting_torchscript_in_torchscript(
            dst_worker_name: str
        ):
            args = ()
            kwargs = {}
            fut = rpc.api.rpc_async(dst_worker_name, nonexisting_script, args, kwargs)
            ret = fut.wait()
            return ret

        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function nonexisting_script"
        ):
            ret = rpc_async_call_remote_nonexisting_torchscript_in_torchscript(
                dst_worker_name
            )
            self.assertEqual(ret, 0)


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed rpc package does not support python2"
)
class JitRpcTest(JitRpcAsyncOpTest, RpcAgentTestFixture):
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

    @dist_init
    def test_torchscript_function_exception(self):
        dst_worker_name = "worker{}".format((self.rank + 1) % self.world_size)
        with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
            ret = rpc.rpc_sync(dst_worker_name, one_arg, args=(10, 20))

        with self.assertRaisesRegex(RuntimeError, r"one_arg\(\) expected at most"):
            rref = rpc.remote(dst_worker_name, one_arg, args=(10, 20))

    @dist_init
    def test_torchscript_functions_not_supported(self):
        # Right now _rpc_sync_torchscript does not accept annotated torchscript
        # class name or script module class name or their class method names.
        # But rpc_sync still accepts script class name and run it in
        # the same code path as python call.
        # Currently neither rpc_sync or _rpc_sync_torchscript is allowed to
        # accept script module and script module method.
        n = self.rank + 1
        dst_rank = n % self.world_size
        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function"
        ):
            ret = rpc._rpc_sync_torchscript(
                "worker{}".format(dst_rank), MyScriptClass, args=()
            )
        ret = rpc.rpc_sync("worker{}".format(dst_rank), MyScriptClass, args=())

        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function"
        ):
            ret = rpc._rpc_sync_torchscript(
                "worker{}".format(dst_rank), MyScriptModule, args=(self.rank,)
            )

        with self.assertRaisesRegex(
            RuntimeError, "attempted to get undefined function"
        ):
            ret = rpc._rpc_sync_torchscript(
                "worker{}".format(dst_rank), MyScriptModule(self.rank).forward, args=()
            )
        # Python 3.5 and Python 3.6 throw different error message, the only
        # common word can be greped is "pickle".
        with self.assertRaisesRegex(Exception, "pickle"):
            ret = rpc.rpc_sync(
                "worker{}".format(dst_rank), MyScriptModule(self.rank).forward, args=()
            )

    @dist_init
    def test_rref_as_arg_and_return(self):
        @torch.jit.script
        def rref_to_here(rref_var):
            # type: (RRef[Tensor]) -> Tensor
            return rref_var.to_here()

        @torch.jit.script
        def return_rref(rref_var):
            # type: (RRef[Tensor]) -> RRef[Tensor]
            return rref_var

        n = self.rank + 1
        dst_rank = n % self.world_size
        local_ret = one_arg(torch.ones(2, 2))

        # create rref on current rank
        rref = rpc.remote(
            "worker{}".format(self.rank), one_arg, args=(torch.ones(2, 2),)
        )

        # pass rref to another user in rpc call
        ret = rpc.rpc_sync("worker{}".format(dst_rank), rref_to_here, args=(rref,))
        self.assertEqual(ret, local_ret)

        # return rref in rpc call
        rref1 = rpc.rpc_sync("worker{}".format(dst_rank), return_rref, args=(rref,))
        self.assertEqual(rref1.to_here(), local_ret)

        # pass rref to another user in remote call
        rref2 = rpc.remote("worker{}".format(dst_rank), rref_to_here, args=(rref,))
        self.assertEqual(rref2.to_here(), local_ret)

        # return rref in remote call
        rref3 = rpc.remote("worker{}".format(dst_rank), return_rref, args=(rref,))
        self.assertEqual(rref3.to_here().to_here(), local_ret)

    @dist_init
    def test_remote_script_module(self):
        @torch.jit.ignore
        def my_script_module_init(rank):
            # type: (int) -> MyModuleInterface
            return MyScriptModule(rank)

        @torch.jit.script
        def construct_my_script_module(rank):
            # type: (int) -> MyModuleInterface
            return my_script_module_init(rank)

        @torch.jit.script
        def run_ref_script_module(ref_script_module, t):
            # type: (RRef[MyModuleInterface], Tensor) -> Tensor
            module = ref_script_module.to_here()
            return module.forward() + t

        # TODO, need more investigation
        # there is rref leak when shutting down, suspect it is because
        # ref as arg is passed to pybind boundary, and the ref is not garbage
        # collected by python when calling shutdown()
        import torch.distributed.rpc.api as api

        api._ignore_rref_leak = True

        local_ret = MyScriptModule(self.rank).forward() + torch.ones(self.rank)

        n = self.rank + 1
        dst_rank = n % self.world_size
        remote_ref = rpc.remote(
            "worker{}".format(dst_rank), construct_my_script_module, args=(self.rank,)
        )

        # pass rref arg to owner
        ret = rpc.rpc_sync(
            "worker{}".format(dst_rank),
            run_ref_script_module,
            args=(remote_ref, torch.ones(self.rank)),
        )
        self.assertEqual(ret, local_ret)

    @dist_init
    def test_rref_is_owner(self):
        n = self.rank + 1
        dst_rank = n % self.world_size
        rref_var = rpc_return_rref("worker{}".format(dst_rank))

        @torch.jit.script
        def rref_tensor_is_owner(rref_var):
            # type: (RRef[Tensor]) -> bool
            return rref_var.is_owner()

        res = rref_tensor_is_owner(rref_var)
        self.assertEqual(res, False)

# mypy: allow-untyped-defs


import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
    dist_init,
    wait_until_pending_futures_and_users_flushed,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


@torch.jit.script
def two_args_two_kwargs(
    first_arg,
    second_arg,
    first_kwarg=torch.tensor([3, 3]),
    second_kwarg=torch.tensor([4, 4]),
):
    return first_arg + second_arg + first_kwarg + second_kwarg


@torch.jit.script
def script_rpc_async_call(
    dst_worker_name: str, args: tuple[Tensor, Tensor], kwargs: dict[str, Tensor]
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
    ret = fut.wait()
    return ret


@torch.jit.script
def rpc_async_call_with_timeout(
    dst_worker_name: str,
    args: tuple[Tensor, Tensor],
    kwargs: dict[str, Tensor],
    timeout: float,
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs, timeout)
    ret = fut.wait()
    return ret


@torch.jit.script
def rpc_async_call_with_timeout_future_ret(
    dst_worker_name: str,
    args: tuple[Tensor, Tensor],
    kwargs: dict[str, Tensor],
    timeout: float,
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs, timeout)
    return fut


@torch.jit.script
def rpc_async_call_future_ret(
    dst_worker_name: str, args: tuple[Tensor, Tensor], kwargs: dict[str, Tensor]
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)
    return fut


@torch.jit.script
def rref_to_here(rref_var: RRef[Tensor]) -> Tensor:
    return rref_var.to_here()


@torch.jit.script
def rref_to_here_with_timeout(rref_var: RRef[Tensor], timeout: float) -> Tensor:
    return rref_var.to_here(timeout)


@torch.jit.script
def rpc_async_with_rref_arg(dst_worker_name: str, args: tuple[RRef[Tensor]]) -> Tensor:
    fut = rpc.rpc_async(dst_worker_name, rref_to_here, args)
    ret = fut.wait()
    return ret


class JitFaultyAgentRpcTest(RpcAgentTestFixture):
    """
    Run tests for rpc_async in JIT under the faulty agent test fixture to test
    arbitrary timeouts.
    """

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_timeout_in_torchscript_function(self):
        # Call rpc_async + fut.wait() in torchscript function and ensure that
        # timeout is raised.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        expected_error = self.get_timeout_error_regex()
        # Ensure that we get a timeout if we override the default timeout and
        # the RPC takes longer to execute.
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0.5)

        # Ensure that we timeout if we don't specify a timeout but the default
        # is less than the RPC takes to execute.
        rpc._set_rpc_timeout(0.001)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            script_rpc_async_call(dst_worker_name, args, kwargs)

        # Ensure that we run to completion if zero timeout is specified.
        ret = rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0)
        self.assertEqual(ret, torch.tensor([8, 8]))
        # reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_timeout_in_python(self):
        # Ensures timeouts are raised if we call rpc_async from within a
        # torchscript function, but wait on the future in python.
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        expected_error = self.get_timeout_error_regex()

        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0.5)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure timeout if we don't specify but the default is less than the
        # RPC takes to execute.
        rpc._set_rpc_timeout(0.001)
        fut = rpc_async_call_future_ret(dst_worker_name, args, kwargs)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if zero timeout is specified
        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0)
        result = fut.wait()
        self.assertEqual(result, torch.tensor([8, 8]))
        # reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_remote_timeout_to_here_in_jit(self):
        # Test that calling to_here() in JIT will raise timeout error if
        # rpc.remote failed.
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # Will ensure error handling callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Call to_here() within a ScriptFunction and ensure it raises
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref_to_here(rref)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_RREF_FETCH_CALL": 1})
    def test_rref_to_here_timeout_in_jit(self):
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref_to_here_with_timeout(rref, 0.01)

        rref_to_here_with_timeout(rref, 100)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_rref_timeout_pickle_in_jit(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # Will ensure error handling callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Call RPC with RRef arg in JIT, which will go through JIT pickling and
        # ensure error is raised.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc_async_with_rref_arg(dst_worker, (rref,))

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_rref_timeout_pickle_script_func(self):
        # Similar to above test, but calls python rpc with script function.
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # Will ensure error handling callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Call RPC with script function that takes RRef, ensure timeout during pickling
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc.rpc_sync(dst_worker, rref_to_here, args=(rref,))

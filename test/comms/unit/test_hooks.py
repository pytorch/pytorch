#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.


import torch
import torch.comms
from torch._C._comms import (
    AllReducePreHookArgs,
    BroadcastPreHookArgs,
    OpName,
    RemovableHandle,
)
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


@skipIfTorchDynamo("comms hook tests verify eager hook dispatch, not traced")
class TestHooks(TestCase):
    def _create_comm(self, name: str) -> torch.comms.TorchComm:
        """Create a communicator using the fake backend."""
        return torch.comms.new_comm("fake", torch.device("cpu"), name=name)

    def test_register_pre_hook(self) -> None:
        """Test that pre-hooks are called before collective operations."""
        comm = self._create_comm("test_pre_hook")

        pre_hook_calls: list[OpName] = []

        def my_pre_hook(name: OpName, op_id: int, args) -> None:
            pre_hook_calls.append(name)

        handle = comm.register_pre_hook(my_pre_hook)
        self.assertIsInstance(handle, RemovableHandle)

        # Run a collective operation
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)

        # Verify hook was called
        self.assertEqual(len(pre_hook_calls), 1)
        self.assertEqual(pre_hook_calls[0], OpName.all_reduce)

        comm.finalize()

    def test_register_post_hook(self) -> None:
        """Test that post-hooks are called after collective operations."""
        comm = self._create_comm("test_post_hook")

        post_hook_call_count = 0

        def my_post_hook(op_id: int, args) -> None:
            nonlocal post_hook_call_count
            post_hook_call_count += 1

        handle = comm.register_post_hook(my_post_hook)
        self.assertIsInstance(handle, RemovableHandle)

        # Run a collective operation
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)

        # Verify hook was called
        self.assertEqual(post_hook_call_count, 1)

        comm.finalize()

    def test_hook_op_id_correlation(self) -> None:
        """Test that pre-hook and post-hook op_ids match for the same operation."""
        comm = self._create_comm("test_op_id")

        pre_op_ids: list[int] = []
        post_op_ids: list[int] = []

        def my_pre_hook(name: OpName, op_id: int, args) -> None:
            pre_op_ids.append(op_id)

        def my_post_hook(op_id: int, args) -> None:
            post_op_ids.append(op_id)

        comm.register_pre_hook(my_pre_hook)
        comm.register_post_hook(my_post_hook)

        # Run multiple operations
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)
        comm.barrier(async_op=False)

        # Verify op_ids match and increase
        self.assertEqual(len(pre_op_ids), 2)
        self.assertEqual(len(post_op_ids), 2)
        self.assertEqual(pre_op_ids[0], post_op_ids[0])
        self.assertEqual(pre_op_ids[1], post_op_ids[1])
        self.assertLess(pre_op_ids[0], pre_op_ids[1])

        comm.finalize()

    def test_hook_removal(self) -> None:
        """Test that hooks are not called after removal."""
        comm = self._create_comm("test_removal")

        call_count = 0

        def my_pre_hook(name: OpName, op_id: int, args) -> None:
            nonlocal call_count
            call_count += 1

        handle = comm.register_pre_hook(my_pre_hook)

        # First operation - hook should be called
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)
        self.assertEqual(call_count, 1)

        # Remove the hook
        handle.remove()

        # Second operation - hook should NOT be called
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)
        self.assertEqual(call_count, 1)

        comm.finalize()

    def test_multiple_hooks(self) -> None:
        """Test that multiple hooks are all called."""
        comm = self._create_comm("test_multi")

        hook1_calls = 0
        hook2_calls = 0

        def hook1(name: OpName, op_id: int, args) -> None:
            nonlocal hook1_calls
            hook1_calls += 1

        def hook2(name: OpName, op_id: int, args) -> None:
            nonlocal hook2_calls
            hook2_calls += 1

        comm.register_pre_hook(hook1)
        comm.register_pre_hook(hook2)

        # Run operation
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=False)

        # Both hooks should be called
        self.assertEqual(hook1_calls, 1)
        self.assertEqual(hook2_calls, 1)

        comm.finalize()

    def test_pre_hook_args_typed(self) -> None:
        """Test that pre-hook receives typed per-collective args."""
        comm = self._create_comm("test_args")

        captured_args: list = []

        def my_pre_hook(name: OpName, op_id: int, args) -> None:
            captured_args.append((name, op_id, args))

        comm.register_pre_hook(my_pre_hook)

        tensor = torch.ones(10)
        comm.broadcast(tensor, root=0, async_op=False)

        self.assertEqual(len(captured_args), 1)
        name, op_id, args = captured_args[0]
        self.assertEqual(name, OpName.broadcast)
        self.assertIsInstance(op_id, int)
        self.assertIsInstance(args, BroadcastPreHookArgs)
        self.assertEqual(args.root, 0)
        self.assertFalse(args.async_op)

        comm.finalize()

    def test_pre_hook_args_all_reduce(self) -> None:
        """Test that all_reduce pre-hook receives AllReducePreHookArgs."""
        comm = self._create_comm("test_ar_args")

        captured_args: list = []

        def my_pre_hook(name: OpName, op_id: int, args) -> None:
            captured_args.append(args)

        comm.register_pre_hook(my_pre_hook)

        tensor = torch.ones(10, dtype=torch.float32)
        comm.all_reduce(tensor, torch.comms.ReduceOp.SUM, async_op=True)

        self.assertEqual(len(captured_args), 1)
        args = captured_args[0]
        self.assertIsInstance(args, AllReducePreHookArgs)
        self.assertEqual(args.tensor.numel(), 10)
        self.assertTrue(args.async_op)

        comm.finalize()

    def test_register_abort_hook(self) -> None:
        """Test that abort hooks can be registered."""
        comm = self._create_comm("test_abort")

        abort_called = False

        def my_abort_hook() -> None:
            nonlocal abort_called
            abort_called = True

        handle = comm.register_abort_hook(my_abort_hook)
        self.assertIsInstance(handle, RemovableHandle)

        # We don't trigger abort in this test as it would terminate the process
        # Just verify registration works
        handle.remove()

        comm.finalize()


if __name__ == "__main__":
    run_tests()

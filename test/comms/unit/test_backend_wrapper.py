#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests that BackendWrapper (the c10d::Backend shim around TorchComm) routes
collectives through TorchComm so that pre/post hooks registered on the
underlying TorchComm fire.
"""

import torch
import torch.comms
import torch.distributed as dist
from torch._C._comms import _BackendWrapper, OpName
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


def _make_wrapped_pg(name: str) -> tuple[torch.comms.TorchComm, dist.ProcessGroup]:
    """Create a fake-backend TorchComm wrapped in a ProcessGroup."""
    comm = torch.comms.new_comm("fake", torch.device("cpu"), name=name)
    wrapper = _BackendWrapper(comm)

    dummy_store = dist.HashStore()
    pg = dist.ProcessGroup(dummy_store, comm.get_rank(), comm.get_size())
    pg._register_backend(
        comm.get_device(),
        dist.ProcessGroup.BackendType.CUSTOM,
        # pyre-fixme[6]: _BackendWrapper implements dist.Backend but stubs aren't aware
        wrapper,
    )
    # pyre-fixme[6]: _set_group_name expects GroupName, which is an alias for str
    pg._set_group_name(name)
    return comm, pg


@skipIfTorchDynamo("backend-wrapper hook tests verify eager hook dispatch, not traced")
class TestBackendWrapperHooks(TestCase):
    def test_hooks_fire_when_invoked_through_backend_wrapper(self) -> None:
        """Pre/post hooks on TorchComm fire when collectives go through the wrapper."""
        comm, pg = _make_wrapped_pg("test_hooks_fire")

        pre_ops: list[OpName] = []
        post_count = 0

        def pre_hook(name: OpName, op_id: int, args) -> None:
            pre_ops.append(name)

        def post_hook(op_id: int, args) -> None:
            nonlocal post_count
            post_count += 1

        comm.register_pre_hook(pre_hook)
        comm.register_post_hook(post_hook)

        tensor = torch.ones(10)

        pg.allreduce([tensor]).wait()
        pg.broadcast(tensor, root=0).wait()
        pg.barrier().wait()

        self.assertEqual(pre_ops, [OpName.all_reduce, OpName.broadcast, OpName.barrier])
        self.assertEqual(post_count, 3)

        comm.finalize()

    def test_hook_op_ids_correlate_and_increase(self) -> None:
        """Pre/post op_ids match per call and increase across calls."""
        comm, pg = _make_wrapped_pg("test_op_ids")

        pre_ids: list[int] = []
        post_ids: list[int] = []

        def pre_hook(name: OpName, op_id: int, args) -> None:
            pre_ids.append(op_id)

        def post_hook(op_id: int, args) -> None:
            post_ids.append(op_id)

        comm.register_pre_hook(pre_hook)
        comm.register_post_hook(post_hook)

        tensor = torch.ones(10)
        pg.allreduce([tensor]).wait()
        pg.barrier().wait()

        self.assertEqual(len(pre_ids), 2)
        self.assertEqual(len(post_ids), 2)
        self.assertEqual(pre_ids[0], post_ids[0])
        self.assertEqual(pre_ids[1], post_ids[1])
        self.assertLess(pre_ids[0], pre_ids[1])

        comm.finalize()

    def test_removed_hooks_do_not_fire(self) -> None:
        """After remove(), hooks no longer fire for wrapper-invoked collectives."""
        comm, pg = _make_wrapped_pg("test_removed")

        pre_count = 0
        post_count = 0

        def pre_hook(name: OpName, op_id: int, args) -> None:
            nonlocal pre_count
            pre_count += 1

        def post_hook(op_id: int, args) -> None:
            nonlocal post_count
            post_count += 1

        pre_handle = comm.register_pre_hook(pre_hook)
        post_handle = comm.register_post_hook(post_hook)

        tensor = torch.ones(10)
        pg.allreduce([tensor]).wait()
        self.assertEqual(pre_count, 1)
        self.assertEqual(post_count, 1)

        pre_handle.remove()
        post_handle.remove()

        pg.allreduce([tensor]).wait()
        self.assertEqual(pre_count, 1)
        self.assertEqual(post_count, 1)

        comm.finalize()


if __name__ == "__main__":
    run_tests()

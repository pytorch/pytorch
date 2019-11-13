from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import distributed.rpc.dist_utils as dist_utils
import torch
import torch.distributed.rpc as rpc
from distributed.rpc.process_group.process_group_rpc_agent_mixin import (
    ProcessGroupRpcAgentMixin,
)
from distributed.rpc.rpc_test import RpcTest


def requires_process_group_agent(message=""):
    def decorator(old_func):
        return unittest.skipUnless(
            dist_utils.TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP", message
        )(old_func)

    return decorator


class ProcessGroupRpcTest(ProcessGroupRpcAgentMixin, RpcTest):
    """
        Allowing customization on top of `RpcTest`,
        including skipping and adding test methods
        specifically for `ProcessGroupAgent`.
    """

    @dist_utils.dist_init(setup_model_parallel=False)
    def test_duplicate_name(self):
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            store, _, _ = next(
                torch.distributed.rendezvous(
                    self.init_method, rank=self.rank, world_size=self.world_size
                )
            )
            rpc._init_rpc(
                backend=self.rpc_backend,
                store=store,
                self_name="duplicate_name",
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
            )
        rpc.join_rpc()

    def test_requires_process_group_agent_decorator(self):
        @requires_process_group_agent("test_func did not run")
        def test_func():
            return "expected result"

        self.assertEqual(test_func(), "expected result")

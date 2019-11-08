from __future__ import absolute_import, division, print_function, unicode_literals

import dist_utils
import rpc_test
import torch.distributed.rpc as rpc
from dist_utils import dist_init


def requires_process_group_agent(message=""):
    def decorator(old_func):
        return unittest.skipUnless(
            dist_utils.TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP", message
        )(old_func)

    return decorator


class ProcessGroupRpcTest(rpc_test.RpcTest):
    @dist_init(setup_model_parallel=False)
    def test_duplicate_name(self):
        with self.assertRaisesRegex(RuntimeError, "is not unique"):
            rpc.init_model_parallel(
                self_name="duplicate_name",
                backend=rpc.backend_registry.BackendType[
                    dist_utils.TEST_CONFIG.rpc_backend_name
                ],
                init_method=self.init_method,
                self_rank=self.rank,
                world_size=self.world_size,
                rpc_agent_options=self.rpc_agent_options,
            )
        rpc.join_rpc()

    def test_requires_process_group_agent_decorator(self):
        @requires_process_group_agent("test_func did not run")
        def test_func():
            return "expected result"

        if dist_utils.TEST_CONFIG.rpc_backend_name == "PROCESS_GROUP":
            self.assertEqual(test_func(), "expected result")

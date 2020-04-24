import torch.distributed.rpc as rpc
import torch.testing._internal.dist_utils
import torch.distributed.rpc._testing  # noqa
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)

# The following message types are currently retried in the RREF protocol and
# distributed autograd. Thus only these messages should be tested with the
# Faulty RPC Agent.
retryable_message_types = ["RREF_FORK_REQUEST",
                           "RREF_CHILD_ACCEPT",
                           "RREF_USER_DELETE",
                           "CLEANUP_AUTOGRAD_CONTEXT_REQ"]

class FaultyRpcAgentTestFixture(RpcAgentTestFixture):
    @property
    def rpc_backend(self):
        return rpc.backend_registry.BackendType[
            "FAULTY_PROCESS_GROUP"
        ]

    @property
    def rpc_backend_options(self):
        return rpc.backend_registry.construct_rpc_backend_options(
            self.rpc_backend,
            init_method=self.init_method,
            num_send_recv_threads=8,
            num_fail_sends=3,
            messages_to_fail=retryable_message_types,
        )

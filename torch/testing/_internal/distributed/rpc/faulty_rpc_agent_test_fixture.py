import torch.distributed.rpc as rpc
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
    def retryable_message_types(self):
        return retryable_message_types

    @property
    def num_fail_sends(self):
        return 3

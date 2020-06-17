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

# The following messages incur the corresponding delay in seconds while being
# processed in FaultyProcessGroupAgent's enqueueSend() function.
default_messages_to_delay = {
    "PYTHON_CALL": 1.5,  # Python UDF
    "SCRIPT_CALL": 1.5,  # Script/Builtin
}

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

    @property
    def default_messages_to_delay(self):
        return default_messages_to_delay

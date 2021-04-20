import torch.distributed.rpc as rpc
import torch.distributed.rpc._testing  # noqa: F401
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages_to_fail = retryable_message_types
        self.messages_to_delay = default_messages_to_delay

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
            messages_to_fail=self.messages_to_fail,
            messages_to_delay=self.messages_to_delay,
        )

    def setup_fault_injection(self, faulty_messages, messages_to_delay):
        if faulty_messages is not None:
            self.messages_to_fail = faulty_messages
        if messages_to_delay is not None:
            self.messages_to_delay = messages_to_delay

    def get_shutdown_error_regex(self):
        error_regexes = [
            "Encountered exception in ProcessGroupAgent::enqueueSend",
            "Encountered exception in ProcessGroupAgent::listenLoop()",
            "Exception in thread pool task",
            "Connection reset by peer",
            "Connection closed by peer"
        ]
        return "|".join(["({})".format(error_str) for error_str in error_regexes])

    def get_timeout_error_regex(self):
        return "RPC ran for more than"

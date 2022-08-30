import torch.distributed.rpc as rpc
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)
from torch.testing._internal.common_distributed import (
    tp_transports,
)


class TensorPipeRpcAgentTestFixture(RpcAgentTestFixture):
    @property
    def rpc_backend(self):
        return rpc.backend_registry.BackendType[
            "TENSORPIPE"
        ]

    @property
    def rpc_backend_options(self):
        return rpc.backend_registry.construct_rpc_backend_options(
            self.rpc_backend,
            init_method=self.init_method,
            _transports=tp_transports()
        )

    def get_shutdown_error_regex(self):
        # FIXME Once we consolidate the error messages returned by the
        # TensorPipe agent put some more specific regex here.
        error_regexes = [".*"]
        return "|".join(["({})".format(error_str) for error_str in error_regexes])

    def get_timeout_error_regex(self):
        return "RPC ran for more than"

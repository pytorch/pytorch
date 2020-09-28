import torch.distributed.rpc as rpc
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
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
        )

    def get_shutdown_error_regex(self):
        # FIXME Once we consolidate the error messages returned by the
        # TensorPipe agent put some more specific regex here.
        error_regexes = [".*"]
        return "|".join(["({})".format(error_str) for error_str in error_regexes])

    def get_timeout_error_regex(self):
        return "RPC ran for more than"

    def get_env_vars(self):
        # Set this to the lowest level, to avoid flooding the output. This will
        # print information about the "boundary" between TensorPipe and PyTorch,
        # by logging all the calls that the agent makes into TensorPipe and all
        # the callbacks that are invoked. This will hopefully be enough to
        # figure out whether the issue lies in TensorPipe or in the agent.
        env_vars = super().get_env_vars()
        env_vars["TP_VERBOSE_LOGGING"] = "1"
        return env_vars

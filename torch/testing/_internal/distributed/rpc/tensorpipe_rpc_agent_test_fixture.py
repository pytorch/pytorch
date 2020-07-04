import torch.distributed.rpc as rpc

# In order to run the existing test RPC and Distributed Autograd test suites
# with Tensorpipe, we introduce a new class in both rpc_test.py and
# dist_autograd_test.py that inherits from the base test classes and this
# fixture. In order for this mulitple inheritance to work, this class cannot
# inherit from RpcAgentTestFixture, since this and the base test classes would
# then have a common ancestor (RpcAgentTestFixture), which is not allowed.
class TensorPipeRpcAgentTestFixture(object):
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

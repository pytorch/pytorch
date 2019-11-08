import rpc_test
import torch.distributed.rpc as rpc


class ProcessGroupRpcTest(rpc_test.RpcTest):
    @property
    def rpc_agent_options(self):
        return rpc.backend_registry.construct_rpc_agent_options(
            self.rpc_backend,
            # Use enough 'num_send_recv_threads' until we fix https://github.com/pytorch/pytorch/issues/26359
            num_send_recv_threads=16,
        )

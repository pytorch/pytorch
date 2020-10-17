import torch.distributed.rpc as rpc
from torch.testing._internal.dist_utils import dist_init
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
from torch.testing._internal.distributed.rpc.dist_examples.parameter_server import run_ps

# Tests the tutorials/examples from https://github.com/pytorch/examples/tree/master/distributed
class DistExamplesTest(RpcAgentTestFixture):

    @dist_init(setup_rpc=False)
    def test_batch_updating_parameter_server(self):

        if self.rank != 0:
            rpc.init_rpc(
                f"trainer{self.rank}",
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        else:
            rpc.init_rpc(
                "ps",
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
            run_ps([f"trainer{r}" for r in range(1, self.world_size)])

        rpc.shutdown()

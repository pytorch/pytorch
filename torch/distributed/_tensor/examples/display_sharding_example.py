import itertools
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.distributed._tensor import (
    DeviceMesh,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.input_reshard import input_reshard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    ModelArgs,
    NUM_DEVICES,
    skip_unless_torch_gpu,
    Transformer,
    with_comms,
)

class DisplayShardingExampleTest():
    def single_function(self, is_seq_parallel=False, recompute_activation=False):
            inp_size = [8, 10]
            # Ensure all tp ranks have same input.
            rng_seed = self.rank if is_seq_parallel else 0
            torch.manual_seed(rng_seed)
            inp = torch.rand(*inp_size, device=self.device_type)
            model = MLPModule(self.device_type)

            LR = 0.25

            optim = torch.optim.SGD(model.parameters(), lr=LR)

            comm_mode = CommDebugMode()
            with comm_mode:
                output = model(inp)
                output.sum().backward()


            optim.step()

            inp = torch.rand(*inp_size, device=self.device_type)
            output = model(inp)


if __name__ == "__main__":
    single_test_instance = DisplayShardingExampleTest()
    single_test_instance.single_function()

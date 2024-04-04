import torch
from torch.distributed._tensor import DeviceMesh, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    NUM_DEVICES,
    with_comms,
)


c10d_functional = torch.ops.c10d_functional


# simplifed tests from test/distributed/tensor/parallel/test_tp_examples.py
class DistTensorParallelExampleTest(DTensorTestBase):
    @with_comms
    def test_mlp_training(self, is_seq_parallel=False):
        inp_size = [8, 10]
        # Ensure all tp ranks have same input.
        rng_seed = self.rank if is_seq_parallel else 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)
        model = MLPModule(self.device_type)
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(0, NUM_DEVICES),
        )
        parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0))
            if is_seq_parallel
            else ColwiseParallel(),
            "net2": RowwiseParallel(output_layouts=Shard(0))
            if is_seq_parallel
            else RowwiseParallel(),
        }
        model_tp = parallelize_module(model, device_mesh, parallelize_plan)
        output = model_tp(inp)
        model_tp.compile(backend='eager')
        output = model_tp(inp)
        output.sum().backward()


if __name__ == "__main__":
    run_tests()

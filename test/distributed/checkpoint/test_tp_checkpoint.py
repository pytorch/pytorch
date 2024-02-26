# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.distributed.checkpoint as DCP

from torch.distributed._tensor import init_device_mesh

from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class UnevenShardedModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(5, 10, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 15, device=device)
        self.net3 = torch.nn.Linear(15, 1, device=device)

    def forward(self, x):
        return self.net3(self.net2(self.relu(self.net1(x))))


class TestTpCheckpoint(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_tp_checkpoint(self):
        CHECKPOINT_DIR = self.temp_dir
        mesh_shpe = (self.world_size,)
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)

        # create model and move it to GPU with id rank
        model = MLPModule(self.device_type).cuda(self.rank)
        # Parallelize the module based on the given Parallel Style.
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        original_state_dict = deepcopy(model.state_dict())

        DCP.save_state_dict(
            state_dict=original_state_dict,
            storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # Update the parameters so model.state_dict() will be different from original_state_dict.
        torch.manual_seed(0)
        inp = torch.rand(20, 10).cuda(self.rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        state_dict = model.state_dict()

        # ensure the current model parameters are different from original_state_dict before loading from checkpoint
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertNotEqual(param1.to_local(), param2.to_local())

        DCP.load_state_dict(
            state_dict=state_dict,
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )

        # now load from checkpoint to check current model parameters are the same as original_state_dict
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertEqual(param1.to_local(), param2.to_local())

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_tp_checkpoint_load_on_meta_device(self):
        CHECKPOINT_DIR = self.temp_dir
        mesh_shpe = (self.world_size,)
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)

        # create model and move it to GPU with id rank
        model = UnevenShardedModel(self.device_type).cuda(self.rank)
        # Parallelize the module based on the given Parallel Style.
        parallelize_plan = {
            "net1": RowwiseParallel(),
            "net2": ColwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan=parallelize_plan)
        original_state_dict = deepcopy(model.state_dict())

        DCP.save_state_dict(
            state_dict=original_state_dict,
            storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
        )

        model2 = parallelize_module(
            UnevenShardedModel("meta"), tp_mesh, parallelize_plan=parallelize_plan
        )
        state_dict_to_load = model2.state_dict()

        DCP.load_state_dict(
            state_dict=state_dict_to_load,
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
        )
        model2.load_state_dict(state_dict_to_load, assign=True)
        state_dict_after_load = model2.state_dict()

        # After loading, check whether params in state_dict_after_load are equal to original_state_dict.
        for param1, param2 in zip(
            original_state_dict.values(), state_dict_after_load.values()
        ):
            self.assertEqual(param1, param2)


if __name__ == "__main__":
    run_tests()

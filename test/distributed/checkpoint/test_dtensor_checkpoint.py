# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
    skip_unless_torch_gpu,
)
from torch.testing._internal.common_utils import run_tests


class MyModule(torch.nn.Module):
    def __init__(
        self,
        sdt: DTensor,
        rdt: DTensor,
        extra_state: int = 1,
        extra_state_tensor: torch.Tensor = torch.zeros(1),
    ) -> None:
        super().__init__()
        self.rdt = torch.nn.Parameter(rdt)
        self.sdt = torch.nn.Parameter(sdt)
        self._extra_state = extra_state
        self._extra_state_tensor = extra_state_tensor

    @property
    def extra_state(self) -> int:
        return self._extra_state

    @extra_state.setter
    def extra_state(self, new_extra_state: int) -> None:
        self._extra_state = new_extra_state

    @property
    def extra_state_tensor(self) -> torch.Tensor:
        return self._extra_state_tensor

    @extra_state_tensor.setter
    def extra_state_tensor(self, new_extra_state_tensor: torch.Tensor) -> None:
        self._extra_state_tensor = new_extra_state_tensor

    def get_extra_state(self) -> Dict[str, Union[int, torch._tensor.Tensor]]:
        return {
            "extra_state": self._extra_state,
            "extra_state_tensor": self._extra_state_tensor,
        }

    def set_extra_state(
        self, state: Dict[str, Union[int, torch._tensor.Tensor]]
    ) -> None:
        self._extra_state = state["extra_state"]  # pyre-ignore[8]
        self._extra_state_tensor = state["extra_state_tensor"]  # pyre-ignore[8]


class DistributedTensorPlanner(DTensorTestBase):
    @with_comms
    @skip_unless_torch_gpu
    @with_temp_dir
    def test_distributed_tensor_planner(self) -> None:
        CHECKPOINT_DIR = self.temp_dir

        local_tensor = torch.arange(0, 4, dtype=torch.float32)
        local_tensor_2 = torch.arange(4, 8, dtype=torch.float32)
        mesh = DeviceMesh(
            device_type="cuda",
            mesh=range(dist.get_world_size()),
        )

        sharded_dt = distribute_tensor(
            local_tensor, mesh, placements=[Shard(0)]
        )
        replicated_dt = distribute_tensor(
            local_tensor_2, mesh, placements=[Replicate()]
        )
        model = MyModule(sharded_dt, replicated_dt).cuda(dist.get_rank())
        state_dict = model.state_dict()
        """
        When the model is initialized, the state_dict on each rank are as followed:
        rank 0:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([4., 5., 6., 7.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([0.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])), \
                    ('_extra_state', {'extra_state': 1, 'extra_state_tensor': tensor([0.])})
                ]
            )
        rank 1:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([4., 5., 6., 7.], device='cuda:3'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([1.], device='cuda:3'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])),
                    ('_extra_state', {'extra_state': 1, 'extra_state_tensor': tensor([0.])})
                ]
            )
        rank 3:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([4., 5., 6., 7.], device='cuda:2'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([2.], device='cuda:2'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])),
                    ('_extra_state', {'extra_state': 1, 'extra_state_tensor': tensor([0.])})
                ]
            )
        rank 4:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([4., 5., 6., 7.], device='cuda:3'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([3.], device='cuda:3'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])),
                    ('_extra_state', {'extra_state': 1, 'extra_state_tensor': tensor([0.])})
                ]
            )
        """

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(path=CHECKPOINT_DIR),
            planner=dist_cp.DefaultSavePlanner(),
        )
        sharded_dt = distribute_tensor(
            local_tensor * 10, mesh, placements=[Shard(0)]
        )
        replicated_dt = distribute_tensor(
            local_tensor_2 * 10, mesh, placements=[Replicate()]
        )
        model = MyModule(
            sharded_dt,
            replicated_dt,
            extra_state=10,
            extra_state_tensor=torch.ones(1) * 10,
        ).cuda(dist.get_rank())
        state_dict = model.state_dict()
        """
        When the model is re-initialized, we have changed the params in state_dict.
        The updated values are as followed:
        rank 0:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([40., 50., 60., 70.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([0.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])), \
                    ('_extra_state', {'extra_state': 10, 'extra_state_tensor': tensor([10.])})
                ]
            )
        rank 1:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([40., 50., 60., 70.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([10.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])), \
                    ('_extra_state', {'extra_state': 10, 'extra_state_tensor': tensor([10.])})
                ]
            )
        rank 3:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([40., 50., 60., 70.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([20.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])), \
                    ('_extra_state', {'extra_state': 10, 'extra_state_tensor': tensor([10.])})
                ]
            )
        rank 4:
            OrderedDict(
                [
                    ('rdt', DTensor(local_tensor=tensor([40., 50., 60., 70.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Replicate()])),
                    ('sdt', DTensor(local_tensor=tensor([30.], device='cuda:0'), device_mesh=DeviceMesh:([0, 1, 2, 3]), placements=[Shard(dim=0)])), \
                    ('_extra_state', {'extra_state': 10, 'extra_state_tensor': tensor([10.])})
                ]
            )
        """

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=dist_cp.DefaultLoadPlanner(),
        )
        sharded_tensor_dict = {
            0: torch.tensor([0], dtype=torch.float32),
            1: torch.tensor([1], dtype=torch.float32),
            2: torch.tensor([2], dtype=torch.float32),
            3: torch.tensor([3], dtype=torch.float32),
        }
        replicated_tensor = torch.tensor([4, 5, 6, 7], dtype=torch.float32)
        """
        After loading the model from the checkpoint, we want to make sure that the values in state_dict
        match the values that are originally saved to the checkpoint.
        """
        for k, v in state_dict.items():
            if k == "rdt":
                self.assertEqual(replicated_tensor, v.to_local())
            if k == "sdt":
                self.assertEqual(
                    sharded_tensor_dict[dist.get_rank()], v.to_local()
                )
            if k == "_extra_state":
                self.assertEqual(1, v["extra_state"])
                self.assertEqual(torch.tensor([0.0]), v["extra_state_tensor"])


if __name__ == "__main__":
    run_tests()

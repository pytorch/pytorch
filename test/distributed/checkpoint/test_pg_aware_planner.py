# Owner(s): ["oncall: distributed"]
# mypy: ignore-errors

from typing import Dict, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as cp
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_utils import run_tests
from torch.distributed.checkpoint.pg_aware_planner import (
    ProcessGroupAwareSavePlanner,
    ProcessGroupAwareLoadPlanner,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
    skip_unless_torch_gpu,
)


class MyModule(torch.nn.Module):
    def __init__(
        self, rank: int, extra_state: int, extra_state_tensor: torch.Tensor
    ) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(
            torch.arange(
                start=rank * 4, end=rank * 4 + 4, step=1, dtype=torch.float32
            )
        )
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


class TestProcessGroupAwarePlanner(DTensorTestBase):
    def _create_new_dist_group(self):
        world_size = dist.get_world_size()
        group1 = [i for i in range(world_size) if i % 2 == 0]
        group2 = [i for i in range(world_size) if i % 2 != 0]

        # create new fsdp group for resharding
        fsdp_0 = dist.new_group(ranks=group1)
        fsdp_1 = dist.new_group(ranks=group2)
        if dist.get_rank() % 2 == 0:
            my_fsdp = fsdp_0
        else:
            my_fsdp = fsdp_1

        return my_fsdp


    @with_comms
    @skip_unless_torch_gpu
    @with_temp_dir
    def test_process_group_aware_planner(self) -> None:
        CHECKPOINT_DIR = self.temp_dir

        model = MyModule(
            rank=dist.get_rank(),
            extra_state=0,
            extra_state_tensor=torch.tensor([[1.0, -1.0], [1.0, -1.0]]),
        ).cuda(dist.get_rank())

        model = FSDP(model, process_group=self._create_new_dist_group())
        """
        When the model is initialized, the param and extra_state_dict are as followed:
        param: tensor([ 0.,  1., 10., 11.], device='cuda:0', requires_grad=True)
        extra_state: 0
        extra_state: tensor([[ 1., -1.], [ 1., -1.]])
        param: tensor([ 0.,  1., 10., 11.], device='cuda:2', requires_grad=True)
        extra_state:0
        extra_state:tensor([[ 1., -1.], [ 1., -1.]])
        param: tensor([ 4.,  5., 14., 15.], device='cuda:1', requires_grad=True)
        extra_state:0
        extra_state:tensor([[ 1., -1.], [ 1., -1.]])
        param: tensor([ 4.,  5., 14., 15.], device='cuda:3', requires_grad=True)
        extra_state:0
        extra_state:tensor([[ 1., -1.], [ 1., -1.]])
        """

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()

            cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=cp.FileSystemWriter(path=CHECKPOINT_DIR),
                planner=ProcessGroupAwareSavePlanner(),
            )

        model = MyModule(
            rank=100,
            extra_state=100,
            extra_state_tensor=torch.tensor([[100.0, -100.0], [100.0, -100.0]]),
        ).cuda(dist.get_rank())
        model = FSDP(model, process_group=self._create_new_dist_group())
        """
        When the model is re-initialized, we have changed param and extra_state_dict.
        The updated values are as followed:
        param: tensor([400., 401., 402., 403.], device='cuda:0', requires_grad=True)
        extra_state: 100
        extra_state: tensor([[ 100., -100.], [ 100., -100.]])
        param: tensor([400., 401., 402., 403.], device='cuda:2', requires_grad=True)
        extra_state: 100
        extra_state: tensor([[ 100., -100.], [ 100., -100.]])
        param: tensor([400., 401., 402., 403.], device='cuda:1', requires_grad=True)
        extra_state: 100
        extra_state: tensor([[ 100., -100.], [ 100., -100.]])
        param: tensor([400., 401., 402., 403.], device='cuda:3', requires_grad=True)
        extra_state: 100
        extra_state: tensor([[ 100., -100.], [ 100., -100.]])
        """

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            state_dict = model.state_dict()

            cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=cp.FileSystemReader(path=CHECKPOINT_DIR),
                planner=ProcessGroupAwareLoadPlanner(),
            )

            model.load_state_dict(state_dict)

        tensor_dict = {
            0: torch.tensor([0, 1, 10, 11], dtype=torch.float32),
            1: torch.tensor([4, 5, 14, 15], dtype=torch.float32),
            2: torch.tensor([0, 1, 10, 11], dtype=torch.float32),
            3: torch.tensor([4, 5, 14, 15], dtype=torch.float32),
        }

        """
        After loading the model from the checkpoint, we want to make sure that the values of
        param and extra_state_dict match the values that are originally saved to the checkpoint.
        """
        with FSDP.summon_full_params(model):
            self.assertEqual(tensor_dict[dist.get_rank()], model.param.detach())
            self.assertEqual(0, model.extra_state)
            self.assertEqual(
                torch.tensor([[1.0, -1.0], [1.0, -1.0]]),
                model.extra_state_tensor,
            )


if __name__ == "__main__":
    run_tests()

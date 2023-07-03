# Owner(s): ["oncall: distributed"]

import shutil
import tempfile
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


def with_temp_dir(
    func: Optional[Callable] = None,
) -> Optional[Callable]:
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        # Only create temp_dir when rank is 0
        if dist.get_rank() == 0:
            temp_dir = tempfile.mkdtemp()
            print(f"Using temp directory: {temp_dir}")
        else:
            temp_dir = ""
        object_list = [temp_dir]

        # Broadcast temp_dir to all the other ranks
        dist.broadcast_object_list(object_list)
        self.temp_dir = object_list[0]

        try:
            func(self, *args, **kwargs)
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)

    return wrapper


class MyTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestFSSpecNoDist(TestCase):
    def test_fsspec_no_dist(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            state_dict_to_save = MyTestModule().state_dict()

            dcp.save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=FsspecWriter(path),
                no_dist=True,
            )

            state_dict_to_load_to = MyTestModule().state_dict()

            for p1, p2 in zip(
                state_dict_to_save.items(),
                state_dict_to_load_to.items(),
            ):
                self.assertNotEqual(p1, p2)

            # Load from file without any resharding
            dcp.load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=FsspecReader(path),
                no_dist=True,
            )

            for p1, p2 in zip(
                state_dict_to_save.items(),
                state_dict_to_load_to.items(),
            ):
                self.assertEqual(p1, p2)


_STATE_DICT_TYPES = [
    StateDictType.SHARDED_STATE_DICT,
    StateDictType.LOCAL_STATE_DICT,
]


class TestFSSpecWithDist(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    @with_temp_dir
    @parametrize("state_dict_type", _STATE_DICT_TYPES)
    def test_fsspec_with_dist(self, state_dict_type):
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(torch.nn.Linear(8, 8, device="meta"))
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()

        dist.barrier()
        with FSDP.state_dict_type(model, state_dict_type):
            state_dict = {
                "model": model.state_dict(),
            }

            dcp.save_state_dict(
                state_dict=state_dict,
                storage_writer=FsspecWriter(CHECKPOINT_DIR),
                planner=dcp.DefaultSavePlanner(),
            )

        model_2 = FSDP(torch.nn.Linear(8, 8, device="meta"))

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertNotEqual(model.weight, model_2.weight)
                self.assertNotEqual(model.bias, model_2.bias)

        # now load the model and ensure the values are the same
        with FSDP.state_dict_type(model_2, state_dict_type):
            state_dict = {
                "model": model_2.state_dict(),
            }

            dcp.load_state_dict(
                state_dict=state_dict,
                storage_reader=FsspecReader(CHECKPOINT_DIR),
                planner=dcp.DefaultLoadPlanner(),
            )
            model_2.load_state_dict(state_dict["model"])

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                self.assertEqual(model.weight, model_2.weight)
                self.assertEqual(model.bias, model_2.bias)

        dist.barrier()


instantiate_parametrized_tests(TestFSSpecWithDist)
if __name__ == "__main__":
    run_tests()

# Owner(s): ["oncall: distributed"]

import copy
import os
import tempfile
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as DCP
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner
from torch.distributed.checkpoint.storage import StorageReader
from torch.distributed.device_mesh import init_device_mesh
from torch.futures import Future
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Linear(8, 16)
        self.net2 = nn.Linear(16, 32)
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x


def _construct_sharded_state_dict(fp_name: str, world_size: int) -> Dict[str, Any]:
    """
    This step depends on the source checkpoint format. In our case, the
    source checkpoint format is torch.save(), we can only do torch.load() to
    construct the state_dict.
    """

    # Not much we can do to save the memory because of the way we save the
    # source checkpoint. For other checkpoint formats, one can do incremental
    # load if possible to avoid CPU OOM.
    state_dict = torch.load(fp_name, map_location="cpu")
    device_mesh = init_device_mesh("cuda", [world_size])
    placement = [Shard(0)]
    for key, value in state_dict.items():
        # Sharded the tensor into a DTensor with each rank contains a shard.
        if torch.is_tensor(value):
            state_dict[key] = DTensor.from_local(
                value.cuda(), device_mesh, [Replicate()], run_check=False
            ).redistribute(
                placements=placement,
            )
        else:
            raise RuntimeError(
                "We don't support conversion of objects other than tensors. If "
                "this is required, you have to convert the object to a bytes "
                "stream."
            )

    return state_dict


def _offline_convert(fp_name: str, target_dir: str, world_size: int):
    # The definition of offline conversion is to convert the existing
    # checkpoints without running a real trainer to do training or
    # inference.
    #
    # For offline conversion, we only need to create a state_dict with
    # FQN->DTensor mapping and save it with DCP.save().
    state_dict = _construct_sharded_state_dict(fp_name, world_size)
    DCP.save(state_dict, checkpoint_id=target_dir)


def _construct_state_dict_and_metadata(
    fp_name: str, world_size: int
) -> Tuple[Dict[str, Any], Metadata]:
    """
    This step depends on the source checkpoint format. In our case, the
    source checkpoint format is torch.save(), we can only do torch.load() to
    construct the state_dict.
    """

    # TODO: formalize this an public API in DCP, and consider to include the
    # logic in TorchSaveReader.
    # https://github.com/pytorch/pytorch/issues/118207

    # Not much we can do to save the memory because of the way we save the
    # source checkpoint. For other checkpoint formats, one can do incremental
    # load if possible to avoid CPU OOM.
    state_dict = torch.load(fp_name, map_location="cpu")

    state_dict_metadata = {}
    placement = [Shard(0)]
    for key, value in state_dict.items():
        # Sharded the tensor into a DTensor with each rank contains a shard.
        size = value.size()
        state_dict_metadata[key] = TensorStorageMetadata(
            TensorProperties(dtype=value.dtype),
            size,
            [ChunkStorageMetadata(size, torch.Size([0 for _ in size]))],
        )
    return state_dict, Metadata(state_dict_metadata=state_dict_metadata)


class TorchSaveReader(StorageReader):
    def __init__(self, fp_name: str, world_size: int) -> None:
        # This is likely to cause CPU OOM if the model size is large.
        # An alternative solution is to actually load the data in # `read_data()`.
        # The performance may be slow depending on the source checkpointing format.
        state_dict, metadata = _construct_state_dict_and_metadata(fp_name, world_size)
        super().__init__()
        self.state_dict = state_dict
        self.metadata = metadata

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        return

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        for req in plan.items:
            if req.type == LoadItemType.BYTE_IO:
                raise RuntimeError("The conversion does not support BYTE_IO yet.")
            else:
                tensor = self.state_dict[req.storage_index.fqn]
                tensor = narrow_tensor_by_index(
                    tensor, req.storage_offsets, req.lengths
                )
                target_tensor = planner.resolve_tensor(req).detach()
                assert target_tensor.size() == tensor.size(), (
                    f"req {req.storage_index} mismatch sizes, "
                    f"{target_tensor.size()} vs {tensor.size()}"
                )
                target_tensor.copy_(tensor)
                planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        return self.metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        return

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return True


class ConversionReader(StorageReader):
    def __init__(self, fp_name: str, world_size: int) -> None:
        if os.path.isdir(fp_name):
            # This is not a good way to detect if the checkpoint is DCP.
            # But this is good enough for an example.
            self.reader = FileSystemReader(fp_name)
        else:
            self.reader = TorchSaveReader(fp_name, world_size)

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        return

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        return self.reader.read_data(plan, planner)

    def read_metadata(self) -> Metadata:
        return self.reader.read_metadata()

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        return self.reader.set_up_storage_reader(metadata, is_coordinator)

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return self.reader.prepare_local_plan(plan)

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return self.reader.prepare_global_plan(global_plan)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return True


class TestTorchSaveToDCP(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_offline_conversion(self) -> None:
        # The definition of offline conversion is to convert the existing
        # checkpoints without running a real trainer to do training or
        # inference.

        model = TestDummyModel()
        with tempfile.NamedTemporaryFile() as fp:
            if self.rank == 0:
                torch.save(model.state_dict(), fp)
                os.sync()
                objects = [fp.name]
                dist.broadcast_object_list(objects)
            else:
                objects = [None]
                dist.broadcast_object_list(objects)
                os.sync()
            fp_name = objects[0]

            # Convert a torch.save checkpoint to a sharded DCP checkpoint.
            _offline_convert(fp_name, self.temp_dir, self.world_size)

        state_dict = copy.deepcopy(model.state_dict())
        # Load the converted DCP checkpoint back with normal DCP.load.
        DCP.load(state_dict, checkpoint_id=self.temp_dir)
        self.assertEqual(state_dict, model.state_dict())

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_online_conversion(self) -> None:
        # The definition of online conversion is to convert the existing
        # checkpoints when the training is about to use the checkpoints.

        model = TestDummyModel()
        with tempfile.NamedTemporaryFile() as fp:
            if self.rank == 0:
                torch.save(model.state_dict(), fp)
                os.sync()
                objects = [fp.name]
                dist.broadcast_object_list(objects)
            else:
                objects = [None]
                dist.broadcast_object_list(objects)
                os.sync()
            fp_name = objects[0]

            copy_state_dict = copy.deepcopy(model.state_dict())
            state_dict = model.state_dict()
            # Use DCP.load() with ConversionReader to read from either torch.save()
            # checkpoint or DCP checkopint.
            DCP.load(
                state_dict,
                storage_reader=ConversionReader(fp_name, self.world_size),
            )
            model.load_state_dict(state_dict)
            self.assertEqual(copy_state_dict, model.state_dict())

            DCP.save(state_dict, checkpoint_id=self.temp_dir)
            DCP.load(
                state_dict,
                storage_reader=ConversionReader(self.temp_dir, self.world_size),
            )
            model.load_state_dict(state_dict)
            self.assertEqual(copy_state_dict, model.state_dict())


if __name__ == "__main__":
    run_tests()

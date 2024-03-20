import argparse
import os
from enum import Enum
from typing import cast, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    Metadata,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner
from torch.distributed.checkpoint.planner_helpers import _create_chunk_list
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.state_dict_saver import _save_state_dict
from torch.distributed.checkpoint.storage import StorageReader
from torch.futures import Future


__all__ = [
    "dcp_to_torch_save",
    "torch_save_to_dcp",
    "BroadcastingTorchSaveReader",
    "DynamicMetaLoadPlanner",
]


class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model, such as
    when converting a DCP checkpoint into a Torch save file.

    . N.B. `state_dict` must be an empty dictionary when used with this LoadPlanner

    .. warning::
        Because the entire state dict is initialized, It's recommended to only utilize
        this LoadPlanner on a single rank or process to avoid OOM.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        assert not state_dict

        # rebuild the state dict from the metadata
        for k, v in metadata.state_dict_metadata.items():
            if isinstance(v, TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            if k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)
            else:
                state_dict[k] = v

        super().set_up_planner(state_dict, metadata, is_coordinator)


class BroadcastingTorchSaveReader(StorageReader):
    """
    StorageReader for reading a Torch Save file. This reader will read the entire checkpoint
    on the coordinator rank, and then broadcast and shard each tensor to all ranks.

    . N.B. Intended to be used with DynamicMetaLoadPlanner

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def __init__(
        self,
        checkpoint_id: Optional[Union[str, os.PathLike]] = None,
        coordinator_rank: int = 0,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.coordinator_rank = coordinator_rank

    def read_metadata(self) -> Metadata:
        """Extends the default StorageReader to support building the metadata file"""
        # Metadata is built in planner.set_up_planner, since we are not actually reading metadata from
        # the disk
        return Metadata(state_dict_metadata={})

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        """
        Reads torch save data on the coordinator rank, and broadcast afterwards
        this incurrs a communication cost, but avoids having to load
        the entire checkpoint on each rank, hopefully preventing OOM issues
        """
        planner = cast(DefaultLoadPlanner, planner)

        # data is read in on the coordinator rank, and broadcast afterwards
        # this incurrs a communication cost, but it avoids having to load
        # the entire checkpoint on each rank, hopefully preventing OOM issues
        # TODO: read on each host, instead of only the coordinator
        if self.is_coordinator:
            assert self.checkpoint_id is not None
            torch_state_dict = torch.load(self.checkpoint_id, map_location="cpu")
            if planner.flatten_state_dict:
                torch_state_dict, _ = flatten_state_dict(torch_state_dict)
        else:
            torch_state_dict = None

        for req in plan.items:
            if req.type == LoadItemType.BYTE_IO:
                raise RuntimeError(
                    f"Non-tensor value identified at {req.storage_index.fqn}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            #  Broadcast the tensor from the coordinator rank
            if self.is_coordinator:
                tensor = torch_state_dict[req.storage_index.fqn].cuda()
            else:
                tensor = torch.empty_like(planner.state_dict[req.storage_index.fqn])

            dist.broadcast(tensor, src=self.coordinator_rank, async_op=False)

            tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
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

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        """Implementation of the StorageReader method"""
        self.is_coordinator = is_coordinator
        if self.is_coordinator:
            assert dist.get_rank() == self.coordinator_rank

        assert self.checkpoint_id is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        """Implementation of the StorageReader method"""
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        """Implementation of the StorageReader method"""
        return global_plan

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        """Implementation of the StorageReader method"""
        self.checkpoint_id = checkpoint_id

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Implementation of the StorageReader method"""
        return os.path.isfile(checkpoint_id)


class DynamicMetaLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which creates a new Metadata object based on the passed in state dict,
    avoiding the need to read metadata from disk. This is useful when reading formats which don't have a
    metadata file, like Torch Save files.

    . N.B. Intended to be used with BroadcastingTorchSaveReader

    .. warning::
        Current implementation only supports loading Tensors.

    >>> # xdoctest: +SKIP("undefined vars")
    >>> sd = {"mode": model}
    >>> dcp.load(
    >>>    sd,
    >>>    storage_reader=BroadcastingTorchSaveReader(),
    >>>    planner=DynamicMetaLoadPlanner(),
    >>>    checkpoint_id="path_to_model.pt"
    >>> )
    """

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata,
        is_coordinator: bool,
    ) -> None:
        """Setups of the planner, extnding default behavior by creating the Metadata object from the state dict"""
        super().set_up_planner(state_dict, metadata, is_coordinator)

        state_dict_metadata: Dict[str, STORAGE_TYPES] = {}
        for key, tensor in self.state_dict.items():
            if not torch.is_tensor(tensor):
                raise RuntimeError(
                    f"Non-tensor value identified at {key}. "
                    f"At this time {type(self).__name__} only supports loading Tensors."
                )

            state_dict_metadata[key] = TensorStorageMetadata(
                TensorProperties(dtype=tensor.dtype),
                tensor.size(),
                _create_chunk_list(tensor),
            )
        self.metadata = Metadata(state_dict_metadata=state_dict_metadata)


def dcp_to_torch_save(
    dcp_checkpoint_dir: Union[str, os.PathLike],
    torch_save_path: Union[str, os.PathLike],
):
    """
    Given a directory containing a DCP checkpoint, this function will convert it into a
    Torch save file.

    Args:
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.
        torch_save_path: Filename to store the converted Torch save file.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """
    sd: STATE_DICT_TYPE = {}

    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    torch.save(sd, torch_save_path)


def torch_save_to_dcp(
    torch_save_path: Union[str, os.PathLike],
    dcp_checkpoint_dir: Union[str, os.PathLike],
):
    """
    Given the location of a torch save file, converts it into a DCP checkpoint.

    Args:
        torch_save_path: Filename to store the converted Torch save file.
        dcp_checkpoint_dir: Directory containing the DCP checkpoint.

    .. warning::
        To avoid OOM, it's recommended to only run this function on a single rank.
    """

    state_dict = torch.load(torch_save_path)
    # we don't need stateful behavior here because the expectation is anything loaded by
    # torch.load would not contain stateful objects.
    _save_state_dict(
        state_dict, storage_writer=FileSystemWriter(dcp_checkpoint_dir), no_dist=True
    )


if __name__ == "__main__":

    class FormatMode(Enum):
        TORCH_TO_DCP = "torch_to_dcp"
        DCP_TO_TORCH = "dcp_to_torch"

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        help="Conversion mode",
        choices=[m.value for m in FormatMode],
        default=FormatMode.TORCH_TO_DCP,
    )
    parser.add_argument("src", type=str, help="Path to the source model")
    parser.add_argument("dst", type=str, help="Path to the destination model")
    args = parser.parse_args()

    print(
        f"Converting checkpoint from {args.src} to {args.dst} using method: '{args.mode}'"
    )
    checkpoint_missing_warning = (
        f"No checkpoint found at {args.src}. Skipping conversion."
    )
    if args.mode == FormatMode.TORCH_TO_DCP:
        if os.path.isfile(args.src):
            torch_save_to_dcp(args.src, args.dst)
        else:
            print(checkpoint_missing_warning)
    elif args.mode == FormatMode.DCP_TO_TORCH:
        if os.path.isdir(args.src):
            dcp_to_torch_save(args.src, args.dst)
        else:
            print(checkpoint_missing_warning)

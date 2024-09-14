# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import dataclasses
import io
import logging
import operator
from collections import ChainMap
from functools import reduce
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._dedup_save_plans import dedup_save_plans
from torch.distributed.checkpoint._nested_dict import (
    FLATTEN_MAPPING,
    flatten_state_dict,
)
from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors
from torch.distributed.checkpoint._traverse import set_element
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    StorageMeta,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.planner_helpers import (
    _create_default_metadata_only_plan,
    _create_read_items,
    _create_write_items,
    _init_state_dict,
)
from torch.distributed.checkpoint.utils import find_state_dict_object
from torch.distributed.tensor import DTensor

from . import _version


logger: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "DefaultSavePlanner",
    "DefaultLoadPlanner",
    "create_default_local_load_plan",
    "create_default_global_load_plan",
    "create_default_local_save_plan",
    "create_default_global_save_plan",
]


# TODO: Update docstrings for default_planner.py
class DefaultSavePlanner(SavePlanner):
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_replicated_tensors: Optional[bool] = None,
        dedup_save_to_lowest_rank: bool = False,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.mappings = {}
        self.dedup_save_to_lowest_rank = dedup_save_to_lowest_rank
        if dedup_replicated_tensors is not None:
            logger.warning(
                "DefaultSavePlanner's `dedup_replicated_tensors` argument is being "
                "deprecated, and no longer has any effect. Please remove this argument "
                "from your call."
            )

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: Optional[StorageMeta] = None,
        is_coordinator: bool = False,
    ) -> None:
        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)
        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)
        self.state_dict = state_dict
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan

        return self.plan

    def create_global_plan(
        self, all_plans: List[SavePlan]
    ) -> Tuple[List[SavePlan], Metadata]:
        all_plans = dedup_save_plans(all_plans, self.dedup_save_to_lowest_rank)

        global_plan, metadata = create_default_global_save_plan(all_plans)

        if self.flatten_state_dict:
            # | does not work for Python 3.8 or older version.
            # merged_mappings = reduce(
            #     lambda x, y: x | y, (p.planner_data for p in global_plan)
            # )
            planner_data_dict = [p.planner_data for p in global_plan]
            merged_mappings = dict(ChainMap(*planner_data_dict))
            metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

        if not _validate_global_plan(global_plan, metadata):
            raise ValueError("Failed to validate global plan")

        self.global_plan = global_plan
        self.metadata = metadata

        return self.global_plan, self.metadata

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self.plan = new_plan
        return new_plan

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def lookup_object(self, index: MetadataIndex) -> Any:
        """Extension from the planner interface to make it easy to extend the default planner."""
        return find_state_dict_object(self.state_dict, index)

    def transform_object(self, write_item: WriteItem, object: Any):
        """Extension from the planner interface to make it easy to extend the default planner."""
        if write_item.type == WriteItemType.BYTE_IO:
            bytes = io.BytesIO()
            torch.save(object, bytes)
            object = bytes
        return object


class DefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    allow_partial_load: If False, will raise a runtime error if a key is present in state_dict, but not in the checkpoint.
    """

    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        allow_partial_load: bool = False,
    ) -> None:
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.original_state_dict = {}
        self.mappings = {}
        self.allow_partial_load = allow_partial_load

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        _init_state_dict(state_dict)
        self.original_state_dict = state_dict

        if self.flatten_sharded_tensors:
            state_dict = _flatten_sharded_tensors(state_dict)

        if self.flatten_state_dict:
            state_dict, self.mappings = flatten_state_dict(state_dict)

        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None
        if self.flatten_state_dict:
            # To support checkpoints that are saved before v2.4, we have to
            # differentiate if the missing keys are due to old checkpoints.
            # The contracts are:
            # 1. There are 3 cases when we found a missing key.
            #    1.1 Actual missing key, but allow_partial_load is False
            #    1.2 Actual missing key, but allow_partial load is True
            #    1.3 Old checkpoint, but allow_partial_load is False
            #    1.4 Old checkpoint, but allow_partial_load is True
            # 2. If we found a missing key, we first convert the keys back to
            #    the key format of v2.3
            # 3. If the previous missing keys are in the v2.3 keys, we assume
            #    this is a old checkpoint.
            # 4. Pass the state_dict to `create_default_local_load_plan()`,
            #    which has the logic to check missing for allow_partial_load.
            # So for 1.2 and 1.4 cases, we delegate allow_partial_load check to
            # `create_default_local_load_plan()`. The logic here is to determine
            # whether the checkpoint belong to 2.3 (or before) or 2.4 (or after).
            current_keys = set(self.state_dict.keys())
            load_keys = set(self.metadata.state_dict_metadata.keys())
            missing_keys = load_keys - current_keys
            if missing_keys:
                _version._derived_version = "2_3"
                old_state_dict, old_mappings = flatten_state_dict(
                    self.original_state_dict
                )
                old_keys = set(old_state_dict.keys())
                if old_keys & missing_keys:
                    self.state_dict, self.mappings = old_state_dict, old_mappings
                # _derived_version is only used by flatten_state_dict now.
                # Set it back to None so that later we can save to a new version.
                _version._derived_version = None

        return create_default_local_load_plan(
            self.state_dict, self.metadata, not self.allow_partial_load
        )

    def create_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return create_default_global_load_plan(global_plan)

    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan:
        return new_plan

    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None:
        if self.flatten_state_dict:
            set_element(
                self.original_state_dict,
                self.mappings[read_item.dest_index.fqn],
                torch.load(value, weights_only=False),
            )
        else:
            self.state_dict[read_item.dest_index.fqn] = torch.load(
                value, weights_only=False
            )

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        pass

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """Extension from the planner interface to make it easy to extend the default planner."""
        return find_state_dict_object(self.state_dict, index)

    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """Extension from the planner interface to make it easy to extend the default planner."""
        return narrow_tensor_by_index(tensor, read_item.dest_offsets, read_item.lengths)


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

    def __init__(self, keys=None, *args, **kwargs):
        self.keys = keys
        super().__init__(*args, **kwargs)

    def _should_include_key(self, key: str, metadata: Metadata) -> bool:
        if self.keys is None:
            return True

        if key in self.keys:
            True

        unflattened_keys: List[str] = []
        planner_data = metadata.planner_data.get(key)
        for unflattened_key in planner_data:
            if unflattened_keys:
                unflattened_keys.append(
                    ".".join([unflattened_keys[-1], str(unflattened_key)])
                )

            else:
                unflattened_keys.append(unflattened_key)

        if any(unflattened_key in self.keys for unflattened_key in unflattened_keys):
            return True

        return False

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Optional[Metadata] = None,
        is_coordinator: bool = False,
    ) -> None:
        assert not state_dict
        assert metadata is not None

        # rebuild the state dict from the metadata
        for k, v in metadata.state_dict_metadata.items():
            if not self._should_include_key(k, metadata):
                continue

            if isinstance(v, TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            if k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)
            else:
                state_dict[k] = v

        super().set_up_planner(state_dict, metadata, is_coordinator)


def create_default_local_load_plan(
    state_dict: Dict[str, Any], metadata: Metadata, strict: bool = True
) -> LoadPlan:
    requests = []
    """
    Create the ``LoadPlan`` used by DefaultLoadPlanner.

    It produces one read item per value in ``state_dict`` using the metadata in ``metadata``.

    The default behavior is to match key exactly between state_dict and metadata.
    It handles resharding by issuing multiple read requests against storage in order to match
    load requirements.
    """

    for fqn, obj in state_dict.items():
        # ignore state_dict keys which do not exist in `state_dict` if strict=False
        if fqn not in metadata.state_dict_metadata:
            if strict:
                raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
            else:
                continue

        md = metadata.state_dict_metadata[fqn]
        # Since DTensor supports submesh, adding extra check to ensure _create_read_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)

    return LoadPlan(requests)


def create_default_global_load_plan(
    all_plans: List[LoadPlan],
) -> List[LoadPlan]:
    """
    Create global load plan used by DefaultLoadPlanner.

    The default load behavior involved no global coordination and this function
    currently doesn't change the local plans.
    """
    return all_plans


def create_default_local_save_plan(
    state_dict: Dict[str, Any], is_coordinator: bool
) -> SavePlan:
    """
    Create the ``SavePlan`` used by DefaultSavePlanner.

    On non-coordinator ranks, this function ignores tensors and non-tensor objects,
    only producing writes for ShardedTensor objects.

    On the coordinator rank, produce writes for all values.
    """
    requests = []
    for fqn, obj in state_dict.items():
        # Since DTensor supports submesh, adding extra check to ensure _create_write_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        else:
            # For the plain tensor and non-tensor values, add the request for all
            # the ranks. Coordinator will decides whether to deduplicate the
            # values based on the keys.
            requests += _create_write_items(fqn, obj)

    return SavePlan(requests)


def create_default_global_save_plan(
    all_plans: List[SavePlan],
    rewrite_index_hints: bool = True,
) -> Tuple[List[SavePlan], Metadata]:
    """
    Create the global plan and metadata used by DefaultSavePlanner.

    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.

    The only global planning change is to update index hints in all ``MetadataIndex`` objects if
    ``rewrite_index_hints`` is True.
    """
    md: Dict[str, STORAGE_TYPES] = {}
    new_plans = []
    for plan in all_plans:
        new_items = []
        for item in plan.items:
            if not item.type == WriteItemType.SHARD:
                assert item.index.fqn not in md

            if item.type == WriteItemType.BYTE_IO:
                md[item.index.fqn] = BytesStorageMetadata()
                new_items.append(item)
            else:
                assert item.tensor_data is not None
                tensor_md = cast(
                    TensorStorageMetadata,
                    md.setdefault(
                        item.index.fqn,
                        TensorStorageMetadata(
                            properties=item.tensor_data.properties,
                            size=item.tensor_data.size,
                            chunks=[],
                        ),
                    ),
                )
                new_item = item
                if rewrite_index_hints:
                    new_index = dataclasses.replace(
                        item.index, index=len(tensor_md.chunks)
                    )
                    new_item = dataclasses.replace(item, index=new_index)
                new_items.append(new_item)

                assert (
                    item.tensor_data.chunk is not None
                ), f"""
                    Cannot create MD for tensor without bounds.
                    FQN: {item.index.fqn}
                """
                tensor_md.chunks.append(item.tensor_data.chunk)
        new_plans.append(dataclasses.replace(plan, items=new_items))
    return (new_plans, Metadata(md))


def _create_default_local_metadata(state_dict: STATE_DICT_TYPE) -> Metadata:
    """Return the ``Metadata`` if DefaultSavePlanner was used to checkpoint ``state_dict``."""
    plan = _create_default_metadata_only_plan(state_dict)
    _, md = create_default_global_save_plan([plan])
    return md


def _check_box_overlap(box0: ChunkStorageMetadata, box1: ChunkStorageMetadata) -> bool:
    """Check if two boxes overlap. Tuples are (offset, lengths)."""
    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(box0.offsets)
    for i in range(ndims):
        if box0.offsets[i] >= box1.offsets[i] + box1.sizes[i]:
            return False
        if box1.offsets[i] >= box0.offsets[i] + box0.sizes[i]:
            return False

    return True


def _check_box_bounds(
    outer_box_size: torch.Size, inner_box: ChunkStorageMetadata
) -> bool:
    for i in range(len(outer_box_size)):
        if inner_box.offsets[i] < 0:
            return False
        if inner_box.sizes[i] < 0:
            return False
        if inner_box.offsets[i] + inner_box.sizes[i] > outer_box_size[i]:
            return False

    return True


def _validate_global_plan(global_plan: List[SavePlan], metadata: Metadata) -> bool:
    all_good = True
    for key, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            continue
        if len(value.size) == 0:
            continue
        chunks_volume = 0
        for chunk_idx, chunk0 in enumerate(value.chunks):
            # Compute the volume
            if not _check_box_bounds(value.size, chunk0):
                logger.warning(
                    """
                        key:%s has out of bounds chunk:
                        tensor-size:%s chunk: %s
                    """,
                    key,
                    value.size,
                    chunk0,
                )
                all_good = False
            chunks_volume += reduce(operator.mul, chunk0.sizes, 1)

            # Check for overlap
            for chunk1 in value.chunks[chunk_idx + 1 :]:
                if _check_box_overlap(chunk0, chunk1):
                    logger.warning(
                        "key:%s has overlapping chunks: %s %s", key, chunk0, chunk1
                    )
                    all_good = False

        # Check whether combined chunk cover the whole tensor
        tensor_volume = reduce(operator.mul, value.size, 1)
        if chunks_volume != tensor_volume:
            logger.warning(
                """
                    key:%s invalid fill tensor-volume:
                    %s chunks-volume: %s
                """,
                key,
                tensor_volume,
                chunks_volume,
            )
            all_good = False

    return all_good

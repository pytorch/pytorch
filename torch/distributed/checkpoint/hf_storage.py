# mypy: allow-untyped-defs
import dataclasses
import json
import logging
import queue
from typing import Any, Optional

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint._hf_utils import (
    _gen_file_name,
    _get_dtype,
    _get_safetensors_file_metadata,
    _HFStorageInfo,
    _metadata_fn,
    CUSTOM_METADATA_KEY,
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SHARDED_DIR_NAME,
    SUFFIX,
)
from torch.distributed.checkpoint.filesystem import SerializationFormat
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future


logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["HuggingFaceStorageWriter", "HuggingFaceStorageReader"]


class HuggingFaceStorageWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses Fsspec back-end to communicate with back-end storage.
    Fsspec registration of the storage solution is required.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: Optional[dict[str, int]] = None,
        thread_count: int = 1,
        token: Optional[str] = None,
        save_distributed: bool = False,
        enable_consolidation: bool = False,
        thread_count_consolidation: int = 1,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
                  Needs to have .safetensors files, but can be from any fsspec supported storage,
                  including localFS and hf://.
                  This needs to be a remote path if you want to enable consolidation after saving.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files. If not provided,
                              the tensors will be written to a single file. If none, then all the tensors on the
                              same rank will be written to the same file.
            thread_count: Number of threads to use to write distributed checkpoint. Default to 1.
            token: The token to use to authenticate with huggingface hub.
            save_distributed: If True, save the checkpoint using distributed APIs where every rank saves its own shard.
                        Default is False which assumes rank-0 checkpointing of the full state_dict.
            enable_consolidation: If True, consolidate the sharded checkpoint after saving. The sharded tensors will be
                                saved to path/sharded and the full tensors will be saved to path. Default to False.
            thread_count_consolidation: Number of threads to use for parallel processing of saving data
                                to consolidated output files. Default to 1.
        """

        if token is not None:
            super().__init__(
                path=path,
                token=token,
                serialization_format=SerializationFormat.SAFETENSORS,
                thread_count=thread_count,
            )
        else:
            super().__init__(
                path=path,
                serialization_format=SerializationFormat.SAFETENSORS,
                thread_count=thread_count,
            )
        self.fqn_to_index_mapping: Optional[dict[str, int]] = fqn_to_index_mapping
        self.save_distributed: bool = save_distributed
        self.enable_consolidation: bool = enable_consolidation
        self.consolidated_output_path: Optional[str] = None
        if self.enable_consolidation:
            self.consolidated_output_path = str(self.path)
            self.path = self.fs.concat_path(self.path, SHARDED_DIR_NAME)
        self.thread_count_consolidation = thread_count_consolidation

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        new_plans = []
        for i, plan in enumerate(plans, start=1):
            storage_data: dict[str, Any] = {}
            if self.fqn_to_index_mapping is not None:
                storage_data["fqn_to_index_mapping"] = self.fqn_to_index_mapping
            if self.save_distributed:
                storage_data["shard_index"] = i

            new_plans.append(dataclasses.replace(plan, storage_data=storage_data))

        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        if len(plan.items) == 0:
            fut: Future = Future()
            fut.set_result([])
            return fut

        # storage_plan is a map from key to file index
        storage_data: dict[str, Any] = plan.storage_data
        storage_plan: Optional[dict[str, int]] = None
        shard_index: Optional[int] = None
        if "fqn_to_index_mapping" in storage_data:
            storage_plan = storage_data["fqn_to_index_mapping"]
        if "shard_index" in storage_data:
            shard_index = storage_data["shard_index"]

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(storage_plan.values()) if storage_plan is not None else 1

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = _gen_file_name(file_index, highest_index, shard_index)
            file_queue.put(
                (self.fs.concat_path(self.path, file_name), file_name, write_items)
            )

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        if self.save_distributed and not self.enable_consolidation:
            # if we are saving distributed, without consolidating,
            # then we have no metadata to write because a metadata
            # file with fqn to file mapping doesn't make sense
            # in this case, because fqns will be in multiple files
            logger.info("Not consolidating sharded checkpoint in finish step.")
            return
        if self.save_distributed:
            return consolidate_safetensors_files(
                input_dir=str(self.path),
                output_dir=self.consolidated_output_path,  # type: ignore[arg-type]
                num_threads=self.thread_count_consolidation,
                fqn_to_index_mapping=self.fqn_to_index_mapping,
            )

        # writing a model.index.safetensors.json file with fqn to file mapping
        # for the rank-0 checkpointing case
        metadata_to_write = {}
        storage_md = {}
        total_size = 0
        for wr_list in results:
            storage_md.update(
                {wr.index.fqn: wr.storage_data.relative_path for wr in wr_list}
            )
            total_size += sum([wr.storage_data.length for wr in wr_list])
        metadata_to_write["metadata"] = {"total_size": total_size}
        metadata_to_write["weight_map"] = storage_md

        metadata_path = self.fs.concat_path(self.path, f"{_metadata_fn}")
        with self.fs.create_stream(metadata_path, "w") as metadata_file:
            json.dump(metadata_to_write, metadata_file, indent=2)

    def _split_by_storage_plan(
        self, storage_plan: Optional[dict[str, int]], items: list[WriteItem]
    ) -> dict[int, list[WriteItem]]:
        # storage_plan is a map from key to index
        if storage_plan is None:
            return {1: items}

        buckets = {}
        for item in items:
            key = item.index.fqn

            idx = storage_plan[key]
            if idx not in buckets:
                buckets[idx] = [item]
            else:
                buckets[idx].append(item)

        return buckets

    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class HuggingFaceStorageReader(FsspecReader):
    """
    A reader that reads from a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with storage.
    Fsspec registration of the storage solution is required.
    """

    def __init__(self, path: str, token: Optional[str] = None) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
            Needs to have .safetensors file, but can be from any fsspec supported storage,
            including localFS and hf://.
            token: The token to use to authenticate with huggingface hub.
        """

        if token is not None:
            super().__init__(path=path, token=token)
        else:
            super().__init__(path=path)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            with self.fs.create_stream(file_name, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]

                    stream.seek(item_md.offset)
                    tensor_bytes = stream.read(item_md.length)

                    tensor = torch.frombuffer(
                        tensor_bytes,
                        dtype=item_md.dtype,
                    )
                    tensor = tensor.reshape(item_md.shape)
                    tensor = narrow_tensor_by_index(
                        tensor, req.storage_offsets, req.lengths
                    )
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert target_tensor.size() == tensor.size(), (
                        f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    )

                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(SUFFIX):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with self.fs.create_stream(safetensor_file, "rb") as f:
                safetensors_metadata, metadata_size = _get_safetensors_file_metadata(f)
                custom_metadata = safetensors_metadata.get(DEFAULT_EXTRA_METADATA_KEY)

                dcp_sharding_info = None
                if custom_metadata and custom_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(
                        custom_metadata.get(CUSTOM_METADATA_KEY)
                    )

                for key, val in safetensors_metadata.items():
                    if key == DEFAULT_EXTRA_METADATA_KEY:
                        continue

                    # construct state_dict_metadata
                    if dcp_sharding_info is not None:
                        offset = dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                    else:
                        offset = [0] * len(val[SHAPE_KEY])

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(
                                dtype=_get_dtype(val[DTYPE_KEY])
                            ),
                            size=torch.Size(
                                [
                                    saved + offset
                                    for saved, offset in zip(val[SHAPE_KEY], offset)
                                ]
                            ),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size(offset),
                                    sizes=torch.Size(val[SHAPE_KEY]),
                                )
                            ],
                        )
                    else:
                        state_dict_metadata[key].chunks.append(
                            ChunkStorageMetadata(
                                torch.Size(offset), sizes=torch.Size(val[SHAPE_KEY])
                            )
                        )
                        size = list(state_dict_metadata[key].size)
                        for i in range(len(size)):
                            size[i] = max(size[i], val[SHAPE_KEY][i] + offset[i])
                        state_dict_metadata[key].size = torch.Size(size)

                    # construct storage data
                    if dcp_sharding_info is not None:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                        )
                    else:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=[0] * len(val[SHAPE_KEY])
                        )
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=safetensor_file,
                        offset=val[DATA_OFFSETS_KEY][0] + metadata_size,
                        length=val[DATA_OFFSETS_KEY][1] - val[DATA_OFFSETS_KEY][0],
                        shape=torch.Size(val[SHAPE_KEY]),
                        dtype=_get_dtype(val[DTYPE_KEY]),
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # type: ignore[arg-type]
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id  # type: ignore[union-attr]

        return metadata

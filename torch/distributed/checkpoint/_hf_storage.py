# mypy: allow-untyped-defs
import dataclasses
import io
import json
import queue
import struct
from typing import Any, Optional

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint._hf_planner import _HuggingFaceLoadPlanner
from torch.distributed.checkpoint.filesystem import _StorageInfo, SerializationFormat
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    STORAGE_TYPES,
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


__all__ = ["_HuggingFaceStorageWriter", "_HuggingFaceStorageReader"]

_metadata_fn: str = "model.safetensors.index.json"

FILE_NAME = "model-{cpt_idx}-of-{num_files}"
SHARDED_FILE_NAME = "shard-{shard_idx}-model-{cpt_idx}-of-{num_files}"
SUFFIX = ".safetensors"

# metadata keys
DEFAULT_EXTRA_METADATA_KEY = "__metadata__"
SHAPE_KEY = "shape"
DTYPE_KEY = "dtype"

DTYPE_MAP = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "BF16": torch.bfloat16,
}


class _HuggingFaceStorageWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses Fsspec back-end to communicate with back-end storage.
    Fsspec registration of the storage solution is required.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: Optional[dict[str, int]] = None,
        token: Optional[str] = None,
        save_sharded: bool = False,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
                  Needs to have .safetensors files, but can be from any fsspec supported storage,
                  including localFS and hf://.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files. If not provided,
                              the tensors will be written to a single file. If none, then all the tensors on the
                              same rank will be written to the same file.
            token: The token to use to authenticate with huggingface hub.
            save_sharded: If True, save the checkpoint as a sharded checkpoint where every rank saves its own shard.
                        Default is False which assumes full tensors are being saved.

        """

        if token is not None:
            super().__init__(
                path=path,
                token=token,
                serialization_format=SerializationFormat.SAFETENSORS,
            )
        else:
            super().__init__(
                path=path,
                serialization_format=SerializationFormat.SAFETENSORS,
            )
        self._fqn_to_index_mapping: Optional[dict[str, int]] = fqn_to_index_mapping
        self._save_sharded = save_sharded

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        new_plans = []
        for i, plan in enumerate(plans, start=1):
            storage_data: dict[str, Any] = {}
            if self._fqn_to_index_mapping is not None:
                storage_data["fqn_to_index_mapping"] = self._fqn_to_index_mapping
            if self._save_sharded:
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
            file_name = self._gen_file_name(file_index, highest_index, shard_index)
            file_queue.put(
                (self.fs.concat_path(self.path, file_name), file_name, write_items)
            )

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        if self._save_sharded:
            return

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

    def _gen_file_name(
        self, index: int, largest_index: int, shard_index: Optional[int]
    ) -> str:
        if shard_index is not None:
            return (
                SHARDED_FILE_NAME.format(
                    shard_idx=f"{shard_index}".zfill(5),
                    cpt_idx=f"{index}".zfill(5),
                    num_files=f"{largest_index}".zfill(5),
                )
                + SUFFIX
            )
        else:
            return (
                FILE_NAME.format(
                    cpt_idx=f"{index}".zfill(5), num_files=f"{largest_index}".zfill(5)
                )
                + SUFFIX
            )

    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class _HuggingFaceStorageReader(FsspecReader):
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
        from safetensors import deserialize  # type: ignore[import-not-found]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index.fqn]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            with self.fs.create_stream(file_name, "rb") as stream:
                # TODO: make this more efficient by doing offset reads instead of a
                # full deserialization of the file

                deserialized = deserialize(stream.read())
                deserialized_dict: dict[str, dict[str, Any]] = {
                    tensor_info[0]: tensor_info[1] for tensor_info in deserialized
                }

                for req in reqs:
                    tensor_bytes = deserialized_dict[req.dest_index.fqn]["data"]
                    planner_metadata = planner.metadata  # type: ignore[attr-defined]
                    tensor = torch.frombuffer(
                        tensor_bytes,
                        dtype=planner_metadata.state_dict_metadata[
                            req.dest_index.fqn
                        ].properties.dtype,
                    )
                    # TODO: update this to req.lengths once I get rid of allow_tensor_resize,
                    # shouldn't need to look at the deserialized
                    # dict for metadata as we've already done that in read_metadata file
                    tensor = tensor.reshape(
                        deserialized_dict[req.dest_index.fqn]["shape"]
                    )

                    if (
                        isinstance(planner, _HuggingFaceLoadPlanner)
                        and planner.allow_tensor_resize
                    ):
                        # this is to support the case when users are calling load on
                        # an empty state dict without specifying the correct size of the tensors
                        # in the state dict. Resizing is a hacky way to support this use case.
                        # But will migrate users to _load_state_dict_from_keys method and deprecate this.
                        target_tensor = planner.resolve_tensor(req)
                        target_tensor.resize_(tensor.size())
                        target_tensor = target_tensor.detach()
                    else:
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
        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        storage_data: dict[str, _StorageInfo] = {}

        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(SUFFIX):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with self.fs.create_stream(safetensor_file, "rb") as f:
                metadata = _get_safetensors_file_metadata(f)

                for key, val in metadata.items():
                    state_dict_metadata[key] = TensorStorageMetadata(
                        properties=TensorProperties(dtype=_get_dtype(val[DTYPE_KEY])),
                        size=torch.Size(val[SHAPE_KEY]),
                        chunks=[
                            ChunkStorageMetadata(
                                offsets=torch.Size([0] * len(val[SHAPE_KEY])),
                                sizes=torch.Size(val[SHAPE_KEY]),
                            )
                        ],
                    )

                for key, val in metadata.items():
                    if key == DEFAULT_EXTRA_METADATA_KEY:
                        continue

                    storage_data[key] = _StorageInfo(
                        safetensor_file,
                        val["data_offsets"][0],
                        val["data_offsets"][1] - val["data_offsets"][0],
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata


def _get_safetensors_file_metadata(file_bytes: io.IOBase) -> Any:
    # this uses the same logic that's done in HF code base
    # https://github.com/2404589803/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L5308
    # and follows their documentation on how their files are serialized
    # https://huggingface.co/docs/safetensors/index#format

    header_len_bytes = file_bytes.read(8)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return metadata


def _get_dtype(dtype_str: str) -> torch.dtype:
    try:
        dtype = DTYPE_MAP[dtype_str]
    except KeyError:
        dtype = torch.get_default_dtype()

    return dtype

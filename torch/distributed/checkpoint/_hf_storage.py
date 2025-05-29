# mypy: allow-untyped-defs
import dataclasses
import io
import json
import os
import queue
import struct
from typing import Optional

import fsspec  # type: ignore[import-untyped]

from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
from torch.distributed.checkpoint._hf_planner import (
    _FqnToFileMapping,
    _HuggingFaceLoadPlanner,
)
from torch.distributed.checkpoint.filesystem import SerializationFormat
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    STORAGE_TYPES,
    StorageMeta,
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

FILE_NAME = "model-{cpt_idx}-of-{num_shards}"
SUFFIX = ".safetensors"


class _HuggingFaceStorageWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with the huggingface hub.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: dict[str, int],
        token: Optional[str] = None,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: hf directory where the checkpoint will be written to. Should begin with hf://.
            token: The token to use to authenticate with huggingface hub.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files.

        """
        from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]

        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

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
        self._fqn_to_index_mapping: dict[str, int] = fqn_to_index_mapping

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        plan = super().prepare_local_plan(plan)
        return dataclasses.replace(
            plan, storage_data=_FqnToFileMapping(self._fqn_to_index_mapping)
        )

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        return plans

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
        storage_plan: dict[str, int] = plan.storage_data.fqn_to_file_index_mapping

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(storage_plan.values())

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = self._gen_file_name(file_index, highest_index)
            file_queue.put(
                (self.fs.concat_path(self.path, file_name), file_name, write_items)
            )

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
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
        self, storage_plan: dict[str, int], items: list[WriteItem]
    ) -> dict[int, list[WriteItem]]:
        # storage_plan is a map from key to index
        buckets = {}
        for item in items:
            key = item.index.fqn
            idx = storage_plan[key]
            if idx not in buckets:
                buckets[idx] = [item]
            else:
                buckets[idx].append(item)

        return buckets

    def _gen_file_name(self, index: int, largest_index: int) -> str:
        return (
            FILE_NAME.format(
                cpt_idx=f"{index}".zfill(5), num_shards=f"{largest_index}".zfill(5)
            )
            + SUFFIX
        )

    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class _HuggingFaceStorageReader(FsspecReader):
    """
    A reader that reads from a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with the huggingface hub.
    """

    def __init__(self, path: str, token: Optional[str] = None) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from. Should begin with hf://.
            token: The token to use to authenticate with huggingface hub.
        """
        from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]

        if HfFileSystem.protocol not in fsspec.available_protocols():
            fsspec.register_implementation(HfFileSystem.protocol, HfFileSystem)

        if token is not None:
            super().__init__(path=path, token=token)
        else:
            super().__init__(path=path)

        self.storage_data: dict[str, str] = {}

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from safetensors.torch import load  # type: ignore[import-not-found]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            file_name = self.storage_data[read_item.storage_index.fqn]
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, file_name)
            with self.fs.create_stream(new_path, "rb") as stream:
                loaded_tensors = load(stream.read())
                for req in reqs:
                    tensor = loaded_tensors[req.dest_index.fqn]

                    target_tensor = planner.resolve_tensor(req)
                    if (
                        isinstance(planner, _HuggingFaceLoadPlanner)
                        and planner.allow_tensor_resize
                    ):
                        target_tensor.resize_(tensor.size())
                    else:
                        assert target_tensor.size() == tensor.size(), (
                            f"Tensor size mismatch for {req.dest_index.fqn}: {target_tensor.size()} != {tensor.size()}"
                        )
                    target_tensor = target_tensor.detach()
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        metadata_path = self.fs.concat_path(self.path, _metadata_fn)

        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        storage_data: dict[str, str] = {}

        if not self.fs.exists(metadata_path):
            # if metadata file doesn't exist, create it from the safetensors file
            safetensors_files = []
            for file in self.fs.ls(self.path):
                if file.endswith(SUFFIX):
                    safetensors_files.append(file)

            if len(safetensors_files) != 1:
                raise ValueError(
                    f"Need exactly one safetensors file to load without metadata, found {len(safetensors_files)} files"
                )
            storage_data = {}
            with self.fs.create_stream(safetensors_files[0], "rb") as f:
                keys = _get_safetensors_file_keys(f)

            for key in keys:
                state_dict_metadata[key] = BytesStorageMetadata()
                storage_data[key] = os.path.basename(safetensors_files[0])
        else:
            with self.fs.create_stream(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)

            for key in metadata["weight_map"].keys():
                state_dict_metadata[key] = BytesStorageMetadata()
            storage_data = metadata["weight_map"]

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata


def _get_safetensors_file_keys(file_bytes: io.IOBase) -> list[str]:
    # this uses the same logic that's done in HF code base
    # https://github.com/2404589803/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py#L5308
    # and follows their documentation on how their files are serialized
    # https://huggingface.co/docs/safetensors/index#format

    header_len_bytes = file_bytes.read(8)
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_json = file_bytes.read(header_len)
    metadata = json.loads(header_json)
    return list(metadata.keys())

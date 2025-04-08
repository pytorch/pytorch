# mypy: allow-untyped-defs
import dataclasses
import json
import queue
from typing import Optional

import fsspec  # type: ignore[import-untyped]

from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter
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

        super().__init__(path=path, token=token)
        self._fqn_to_index_mapping: dict[str, int] = fqn_to_index_mapping

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        super().prepare_local_plan(plan)
        return dataclasses.replace(plan, storage_data=self._fqn_to_index_mapping)

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        assert len(plans) == 1, "distributed checkpointing is not yet supported"
        return plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        # storage_plan is a map from key to file index
        storage_plan: dict[str, int] = plan.storage_data

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(buckets.keys())

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = self._gen_file_name(file_index, highest_index)
            file_queue.put(
                (self.fs.concat_path(self.path, file_name), file_name, write_items)
            )

        return super()._write_data(planner, file_queue, safe_tensors=True)

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
        super().__init__(path=path, token=token)
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

                    target_tensor = planner.resolve_tensor(req).detach()
                    target_tensor.resize_(tensor.size())
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        path = self.fs.concat_path(self.path, _metadata_fn)
        with self.fs.create_stream(path, "r") as metadata_file:
            metadata = json.load(metadata_file)

        state_dict_metadata: dict[str, STORAGE_TYPES] = {}
        for key in metadata["weight_map"].keys():
            state_dict_metadata[key] = BytesStorageMetadata()
        metadata = Metadata(
            state_dict_metadata=state_dict_metadata, storage_data=metadata["weight_map"]
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata

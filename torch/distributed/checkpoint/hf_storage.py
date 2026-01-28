# mypy: allow-untyped-defs
import dataclasses
import json
import logging
import queue
import threading
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    consolidate_safetensors_files,
)
from torch.distributed.checkpoint._hf_utils import (
    _gen_file_name,
    _HFStorageInfo,
    _metadata_fn,
    CUSTOM_METADATA_KEY,
    SAVED_OFFSETS_KEY,
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


class HuggingFaceStorageWriter(FileSystemWriter):
    """
    A writer that writes to storage in the huggingface safetensors format.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: dict[str, int] | None = None,
        thread_count: int = 1,
        save_distributed: bool = False,
        enable_consolidation: bool = False,
        thread_count_consolidation: int = 1,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: directory where the checkpoint will be read from.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files. If not provided,
                              the tensors will be written to a single file. If none, then all the tensors on the
                              same rank will be written to the same file.
            thread_count: Number of threads to use to write distributed checkpoint. Default to 1.
            save_distributed: If True, save the checkpoint using distributed APIs where every rank saves its own shard.
                        Default is False which assumes rank-0 checkpointing of the full state_dict.
            enable_consolidation: If True, consolidate the sharded checkpoint after saving. The sharded tensors will be
                                saved to path/sharded and the full tensors will be saved to path. Default to False.
            thread_count_consolidation: Number of threads to use for parallel processing of saving data
                                to consolidated output files. Default to 1.
        """

        super().__init__(
            path=path,
            serialization_format=SerializationFormat.SAFETENSORS,
            thread_count=thread_count,
        )
        self.fqn_to_index_mapping: dict[str, int] | None = fqn_to_index_mapping
        self.save_distributed: bool = save_distributed
        self.enable_consolidation: bool = enable_consolidation
        self.consolidated_output_path: str | None = None
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
        storage_plan: dict[str, int] | None = None
        shard_index: int | None = None
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
            fqn_to_index_mapping: dict[str, int] = (
                self.fqn_to_index_mapping
                if self.fqn_to_index_mapping is not None
                else dict.fromkeys(metadata.state_dict_metadata.keys(), 1)
            )

            return consolidate_safetensors_files(
                input_dir=str(self.path),
                output_dir=self.consolidated_output_path,  # type: ignore[arg-type]
                num_threads=self.thread_count_consolidation,
                fqn_to_index_mapping=fqn_to_index_mapping,
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
        self, storage_plan: dict[str, int] | None, items: list[WriteItem]
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


class HuggingFaceStorageReader(FileSystemReader):
    """
    A reader that reads a checkpoint in the huggingface safetensors format.
    """

    def __init__(self, path: str, thread_count: int = 1) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: directory where the checkpoint will be read from.
            thread_count: Number of threads to use to read distributed checkpoint. Default to 1.
        """

        super().__init__(path=path)
        self.thread_count = thread_count

    def _process_read_request(self, f, req: ReadItem, planner: LoadPlanner) -> None:
        """Helper function to process a single read request."""
        # Create slices for each dimension based on offsets and lengths
        slices = tuple(
            slice(offset, offset + length)
            for offset, length in zip(req.storage_offsets, req.lengths)
        )
        tensor = f.get_slice(req.storage_index.fqn)[slices]
        target_tensor = planner.resolve_tensor(req).detach()

        if target_tensor.size() != tensor.size():
            raise AssertionError(
                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
            )

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)

    def _read_files_from_queue(
        self,
        file_queue: queue.Queue,
        result_queue: queue.Queue,
        planner: LoadPlanner,
    ) -> None:
        from safetensors import safe_open  # type: ignore[import]

        try:
            while True:
                file_name, reqs = file_queue.get_nowait()
                with safe_open(filename=file_name, framework="pt") as f:
                    for req in reqs:
                        self._process_read_request(f, req, planner)
                result_queue.put(True)  # Signal that this file has been processed
        except queue.Empty:
            pass

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from safetensors import safe_open  # type: ignore[import]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        if self.thread_count <= 1 or len(per_file) <= 1:
            for file_name, reqs in per_file.items():
                with safe_open(filename=file_name, framework="pt") as f:
                    for req in reqs:
                        self._process_read_request(f, req, planner)
        else:
            # Use parallel implementation with thread pool
            file_queue: queue.Queue = queue.Queue()
            result_queue: queue.Queue = queue.Queue()

            # Fill the queue with files to process
            for file_name, reqs in per_file.items():
                file_queue.put((file_name, reqs))

            # Create and start worker threads
            threads = []
            num_threads = min(self.thread_count, len(per_file))
            for _ in range(num_threads):
                t = threading.Thread(
                    target=self._read_files_from_queue,
                    args=(file_queue, result_queue, planner),
                )
                t.start()
                threads.append(t)

            # Wait for all threads to complete
            for t in threads:
                t.join()

            # Check if all files were processed
            processed_count = 0
            try:
                while True:
                    result_queue.get_nowait()
                    processed_count += 1
            except queue.Empty:
                pass

            if processed_count != len(per_file):
                raise AssertionError(
                    f"Not all files were processed: {processed_count} out of {len(per_file)}"
                )

        fut: Future = Future()
        fut.set_result(None)
        return fut

    # pyrefly: ignore [bad-override]
    def read_metadata(self) -> Metadata:
        from safetensors import safe_open  # type: ignore[import]
        from safetensors.torch import _getdtype  # type: ignore[import]

        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(SUFFIX):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with safe_open(safetensor_file, framework="pt") as f:
                keys = f.keys()
                extra_metadata = f.metadata()

                dcp_sharding_info = None
                if extra_metadata and extra_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(
                        extra_metadata.get(CUSTOM_METADATA_KEY)
                    )

                for key in keys:
                    shape = f.get_slice(key).get_shape()
                    dtype = f.get_slice(key).get_dtype()
                    # construct state_dict_metadata
                    if dcp_sharding_info is not None:
                        offset = dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                    else:
                        offset = [0] * len(shape)

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=_getdtype(dtype)),
                            size=torch.Size(
                                [saved + offset for saved, offset in zip(shape, offset)]
                            ),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size(offset),
                                    sizes=torch.Size(shape),
                                )
                            ],
                        )
                    else:
                        state_dict_metadata[key].chunks.append(
                            ChunkStorageMetadata(
                                torch.Size(offset), sizes=torch.Size(shape)
                            )
                        )
                        size = list(state_dict_metadata[key].size)
                        for i in range(len(size)):
                            size[i] = max(size[i], shape[i] + offset[i])
                        state_dict_metadata[key].size = torch.Size(size)

                    # construct storage data
                    if dcp_sharding_info is not None:
                        metadata_index = MetadataIndex(
                            fqn=key, offset=dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                        )
                    else:
                        metadata_index = MetadataIndex(fqn=key, offset=[0] * len(shape))
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=safetensor_file,
                        shape=torch.Size(shape),
                        dtype=_getdtype(dtype),
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # type: ignore[arg-type]
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id  # type: ignore[union-attr]

        return metadata

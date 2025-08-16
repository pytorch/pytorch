# mypy: allow-untyped-defs
import dataclasses
import json
import logging
import queue
from typing import Any, Optional

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

__all__ = ["HuggingFaceStorageWriter", "HuggingFaceStorageReader", "QuantizedHuggingFaceStorageReader"]


class HuggingFaceStorageWriter(FileSystemWriter):
    """
    A writer that writes to storage in the huggingface safetensors format.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: Optional[dict[str, int]] = None,
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


class HuggingFaceStorageReader(FileSystemReader):
    """
    A reader that reads a checkpoint in the huggingface safetensors format.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: directory where the checkpoint will be read from.
        """

        super().__init__(path=path)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from safetensors import safe_open  # type: ignore[import]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            with safe_open(filename=file_name, framework="pt") as f:
                for req in reqs:
                    # Create slices for each dimension based on offsets and lengths
                    slices = tuple(
                        slice(offset, offset + length)
                        for offset, length in zip(req.storage_offsets, req.lengths)
                    )
                    tensor = f.get_slice(req.storage_index.fqn)[slices]
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


class QuantizedHuggingFaceStorageReader(HuggingFaceStorageReader):
    """
    Extension of HuggingFaceStorageReader that handles fp8 quantized tensors.
    
    This reader handles the dequantization of fp8 tensors during the read process,
    converting them from quantized blocks to full dequantized tensors before
    copying to the target tensor.
    """

    def __init__(self, path: str, block_size: Optional[int] = None):
        """
        Initialize the FP8 HuggingFace storage reader.
        
        Args:
            path: directory where the checkpoint will be read from
            block_size: optional fixed block size for FP8 dequantization. If None, 
                       block size will be calculated dynamically based on tensor shapes.
        """
        super().__init__(path)
        self.checkpoint_path = path
        self.block_size = block_size
        self.weight_scale_mapping = {}
        self.scale_tensor_cache = {}
        self._load_fp8_metadata()

    def _load_fp8_metadata(self):
        """Load FP8 quantization metadata from checkpoint."""
        try:
            import json
            from pathlib import Path
            
            checkpoint_path = Path(self.checkpoint_path)
            
            # Load weight mapping from index file
            index_files = list(checkpoint_path.glob("*.index.json"))
            if index_files:
                with open(index_files[0], 'r') as f:
                    index_data = json.load(f)
                    weight_map = index_data.get('weight_map', {})
                    self._analyze_weight_scale_mapping(weight_map)
            
        except Exception as e:
            logger.warning(f"Failed to load FP8 metadata: {e}")

    def _analyze_weight_scale_mapping(self, weight_map: dict[str, str]):
        """Analyze weight-scale tensor pairs from weight mapping."""
        for tensor_name in weight_map.keys():
            if tensor_name.endswith('.weight_scale_inv'):
                weight_name = tensor_name.replace('.weight_scale_inv', '.weight')
                if weight_name in weight_map:
                    self.weight_scale_mapping[weight_name] = tensor_name

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        from safetensors import safe_open  # type: ignore[import]

        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            with safe_open(filename=file_name, framework="pt") as f:
                for req in reqs:
                    tensor_fqn = req.storage_index.fqn
                    
                    # Check if this is an FP8 tensor that needs special handling
                    if self._is_fp8_tensor(tensor_fqn, f):
                        tensor = self._read_fp8_tensor_with_block_alignment(req, f)
                    else:
                        # Standard tensor reading
                        slices = tuple(
                            slice(offset, offset + length)
                            for offset, length in zip(req.storage_offsets, req.lengths)
                        )
                        tensor = f.get_slice(tensor_fqn)[slices]
                    
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert target_tensor.size() == tensor.size(), (
                        f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    )

                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def _calculate_scale_shape(self, weight: torch.Tensor, block_size: int) -> tuple[int, int]:
        """Calculate expected scale tensor shape based on weight tensor and block size."""
        rows, cols = weight.shape
        block_rows = (rows + block_size - 1) // block_size  # Ceiling division
        block_cols = (cols + block_size - 1) // block_size  # Ceiling division
        return (block_rows, block_cols)

    def _dequantize_from_fp8(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        dtype=torch.float32,
        block_size: int = 128,
    ) -> torch.Tensor:
        """
        Dequantize FP8 tensor using block-wise scaling.
        
        Args:
            weight: FP8 quantized weight tensor
            scale_inv: Scale inverse tensor for dequantization
            dtype: Target dtype for dequantized tensor
            block_size: Block size for dequantization
            
        Returns:
            Dequantized tensor
        """
        # Convert to float32 for computation
        float_weight = weight.to(torch.float32)
        # Get original dimensions
        orig_shape = weight.shape

        # Calculate block dimensions for the local shard
        expected_scale_shape = self._calculate_scale_shape(weight, block_size)
        block_rows, block_cols = expected_scale_shape

        # NOTE: When processing large models on-the-fly, misalignment between block boundaries
        # and DTensor local shape partitioning can lead to silent numerical inaccuracies.
        dequantized = float_weight.detach().clone().to(dtype=dtype)

        # Apply scaling factors to each block
        for i in range(block_rows):
            row_start = i * block_size
            row_end = min(row_start + block_size, orig_shape[0])

            for j in range(block_cols):
                col_start = j * block_size
                col_end = min(col_start + block_size, orig_shape[1])

                # Get the block
                block = float_weight[row_start:row_end, col_start:col_end]

                scale = scale_inv[i, j]
                block = block * scale

                # Explicitly convert block to dtype
                block_converted = block.to(dtype=torch.float32)
                # Store the dequantized block
                dequantized[row_start:row_end, col_start:col_end] = block_converted

        return dequantized

    def _is_fp8_tensor(self, tensor_fqn: str, safetensor_file) -> bool:
        """
        Check if a tensor is an FP8 quantized tensor that needs dequantization.
        
        Args:
            tensor_fqn: Fully qualified name of the tensor
            safetensor_file: Open safetensors file handle
            
        Returns:
            True if tensor is FP8 and has corresponding scale tensor, False otherwise
        """
        # Skip scale tensors themselves
        if tensor_fqn.endswith('.weight_scale_inv'):
            return False
            
        # Check if this weight tensor has a corresponding scale tensor
        if tensor_fqn not in self.weight_scale_mapping:
            return False
            
        try:
            # Check if the tensor is actually stored in FP8 format
            tensor_slice = safetensor_file.get_slice(tensor_fqn)
            dtype_str = str(tensor_slice.get_dtype())
            
            # Check for FP8 data types
            fp8_dtypes = ['F8_E4M3', 'F8_E5M2', 'float8_e4m3fn', 'float8_e5m2']
            is_fp8_dtype = any(fp8_type in dtype_str for fp8_type in fp8_dtypes)
            
            if is_fp8_dtype:
                logger.debug(f"Detected FP8 tensor: {tensor_fqn} with dtype {dtype_str}")
                return True
                
        except Exception as e:
            logger.warning(f"Failed to check FP8 status for tensor {tensor_fqn}: {e}")
            
        return False

    def _read_fp8_tensor_with_block_alignment(self, req: ReadItem, safetensor_file) -> torch.Tensor:
        """
        Read FP8 tensor with block alignment considerations for FSDP compatibility.
        
        Args:
            req: Read request containing tensor info and required slices
            safetensor_file: Open safetensors file handle
            
        Returns:
            Dequantized tensor ready for use
        """
        tensor_fqn = req.storage_index.fqn
        scale_fqn = self.weight_scale_mapping[tensor_fqn]
        
        try:
            # Load the FP8 weight tensor
            weight_slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(req.storage_offsets, req.lengths)
            )
            fp8_weight = safetensor_file.get_slice(tensor_fqn)[weight_slices]

            # Load the corresponding scale inverse tensor
            # For scale tensors, we typically need the full tensor for proper block alignment
            if scale_fqn not in self.scale_tensor_cache:
                scale_inv = safetensor_file.get_slice(scale_fqn)[:]  # Load full scale tensor
                self.scale_tensor_cache[scale_fqn] = scale_inv
            else:
                scale_inv = self.scale_tensor_cache[scale_fqn]
            
            # Determine block size
            block_size = self.block_size if self.block_size is not None else 128

            # Perform dequantization
            dequantized_tensor = self._dequantize_from_fp8(
                weight=fp8_weight,
                scale_inv=scale_inv,
                dtype=torch.float32,
                block_size=block_size
            )

            return dequantized_tensor

        except Exception as e:
            logger.error(f"Failed to read FP8 tensor {tensor_fqn}: {e}")
            # Fallback to standard tensor reading
            slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(req.storage_offsets, req.lengths)
            )
            return safetensor_file.get_slice(tensor_fqn)[slices]

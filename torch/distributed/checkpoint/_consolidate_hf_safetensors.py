# pyre-strict

import concurrent.futures
import glob
import json
import logging
import math
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import distributed as dist
from torch.distributed.checkpoint._hf_utils import (
    _gen_file_name,
    _get_dcp_custom_metadata,
    _get_safetensors_file_metadata,
    _metadata_fn,
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
)


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _FqnData:
    """
    Dataclass to store information about a tensor (identified by its fully qualified name).

    Attributes:
        offset_in_file: Byte offset where this tensor's data begins in the output file
        shape_in_file: Shape of the tensor in the output file
        dtype_size: Size of the tensor's data type in bytes
        dtype_str: String representation of the tensor's data type
    """

    offset_in_file: int = 0
    shape_in_file: list[int] = field(default_factory=list)
    dtype_size: int = 0
    dtype_str: str = ""


@dataclass
class _OutputFileData:
    """
    Dataclass to store information about an output safetensors file.

    Attributes:
        metadata_size: Size of the metadata section in bytes
        fqn_data: Dictionary mapping tensor names to their metadata
    """

    metadata_size: int = 0
    fqn_data: dict[str, _FqnData] = field(default_factory=dict)


@dataclass
class _InputFileData:
    """
    Dataclass to store information about an input safetensors file.

    Attributes:
        metadata_size: Size of the metadata section in bytes
        metadata: Json metadata from the safetensors file
    """

    metadata_size: int = 0
    metadata: Any = None


def _parse_input_metadata(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Parse metadata from input safetensors files to determine the full tensor shapes and types.

    This function analyzes the metadata from all input files to determine the complete shape
    of each tensor after consolidation. It updates the output_files_data with this information.

    Args:
        input_files_data: dict of metadata from input safetensors files
        output_files_data: Dictionary mapping output file paths to their metadata

    Raises:
        ValueError: If no DCP custom metadata is found in a safetensors file
    """

    from safetensors.torch import _getdtype  # type: ignore[import]

    # Dictionary to track the full size of each tensor across all shards
    fqn_to_size_mapping: dict[str, tuple[list[int], str]] = {}

    for file_data in input_files_data.values():
        safetensors_metadata = file_data.metadata
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
        if not dcp_sharding_info:
            raise ValueError(
                "No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated."
            )

        for key, val in safetensors_metadata.items():
            if key == DEFAULT_EXTRA_METADATA_KEY:
                continue

            # Get the shape of this tensor shard and its offset in the full tensor
            sizes = val[SHAPE_KEY]
            offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]

            if key not in fqn_to_size_mapping:
                # First time seeing this tensor - calculate its full size by adding offsets to dimensions
                cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                fqn_to_size_mapping[key] = (cur_size, val[DTYPE_KEY])
            else:
                # We've seen this tensor before - update its size if this shard extends beyond current known dimensions
                cur_size = fqn_to_size_mapping[key][0]
                for i in range(len(sizes)):
                    cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

    # Now that we know the full size of each tensor, populate the output file data
    for fqn, tensor_info in fqn_to_size_mapping.items():
        tensor_size = tensor_info[0]
        dtype_str = tensor_info[1]
        for output_data in output_files_data.values():
            # Add this tensor to the output file if it's already assigned there
            if fqn in output_data.fqn_data:
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=tensor_size,
                    dtype_size=torch.finfo(_getdtype(dtype_str)).bits
                    // 8,  # Convert bits to bytes
                    dtype_str=dtype_str,
                )


def _write_metadata(
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Write metadata to the beginning of each output safetensors file.

    This function writes the metadata section to each output file, including information
    about tensor shapes, data types, and offsets. It also updates the offset_in_file
    field for each tensor in the output_files_data.

    Args:
        output_files_data: Dictionary mapping output file paths to their metadata
    """
    # Process each output file
    for file_path, output_data in output_files_data.items():
        with open(file_path, "wb") as f:
            metadata = {}
            curr_offset = 0

            # Calculate offsets for each tensor in the file
            for fqn, fqn_data in output_data.fqn_data.items():
                # Calculate the end offset by multiplying all dimensions and the data type size
                end_offset = (
                    curr_offset
                    + math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
                )

                # Store metadata for this tensor
                metadata[fqn] = {
                    SHAPE_KEY: fqn_data.shape_in_file,
                    DTYPE_KEY: fqn_data.dtype_str,
                    DATA_OFFSETS_KEY: [
                        curr_offset,
                        end_offset,
                    ],  # Start and end byte offsets
                }
                # Store the offset for later use when writing the actual tensor data
                fqn_data.offset_in_file = curr_offset

                # Update current offset for the next tensor
                curr_offset = end_offset

            # Convert metadata to JSON and encode as bytes
            json_metadata = json.dumps(metadata)
            json_bytes = json_metadata.encode("utf-8")

            # Write the metadata size as an 8-byte unsigned integer (little-endian)
            size_in_bytes = len(json_bytes)
            header_len = struct.pack("<Q", size_in_bytes)

            # Write the header length and metadata to the file
            f.write(header_len)
            f.write(json_bytes)

            # Store the total metadata size (header + JSON) for later use
            output_data.metadata_size = f.tell()


def _read_tensor_data_mmap(
    file_path: str,
    start_offset: int,
    end_offset: int,
    metadata_size: int,
) -> bytes:
    """
    Read tensor data from a safetensors file using memory mapping for efficiency.

    Args:
        file_path: Path to the safetensors file
        start_offset: Start offset of tensor data within the data section
        end_offset: End offset of tensor data within the data section
        metadata_size: Size of the metadata header

    Returns:
        Raw tensor data as bytes
    """
    # Use mmap for efficient access
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            absolute_start = metadata_size + start_offset
            absolute_end = metadata_size + end_offset
            return bytes(mm[absolute_start:absolute_end])


def _process_output_file(
    output_file: str,
    output_data: _OutputFileData,
    input_files_data: dict[str, _InputFileData],
) -> None:
    """
    Process a single output file by writing tensor data from input files using memory mapping.

    This function is designed to be run in parallel for different output files.

    Args:
        output_file: Path to the output file
        output_data: Metadata for the output file
        input_files_data: Dictionary mapping input file paths to their metadata
    """

    sorted_tensors = sorted(
        output_data.fqn_data.items(), key=lambda x: x[1].offset_in_file
    )

    with open(output_file, "r+b") as output_stream:
        output_stream.seek(0, os.SEEK_END)
        # Process each tensor in sequential output order
        for tensor_fqn, tensor_fqn_data in sorted_tensors:
            full_tensor_mv = memoryview(
                bytearray(
                    math.prod(tensor_fqn_data.shape_in_file)
                    * tensor_fqn_data.dtype_size
                )
            )

            # Process each input safetensors file
            for safetensors_file in input_files_data:
                file_metadata = input_files_data[safetensors_file].metadata
                input_metadata_size = input_files_data[safetensors_file].metadata_size

                if tensor_fqn not in file_metadata:
                    continue

                metadata = file_metadata[tensor_fqn]

                data_offsets = metadata[DATA_OFFSETS_KEY]

                # Use memory mapping to read tensor data efficiently
                data_to_write = _read_tensor_data_mmap(
                    safetensors_file,
                    data_offsets[0],
                    data_offsets[1],
                    input_metadata_size,
                )

                # Get the offsets of this tensor shard within the full tensor
                # pyrefly: ignore [unsupported-operation]
                fqn_custom_metadata = _get_dcp_custom_metadata(file_metadata)[
                    tensor_fqn
                ]  # type: ignore[index]
                offsets_of_tensor_being_read = fqn_custom_metadata[SAVED_OFFSETS_KEY]  # type: ignore[index]

                # Write this tensor shard to the appropriate position in the output file
                _write_sub_tensor_to_file_optimized(
                    full_tensor_mv,
                    data_to_write,
                    tensor_fqn_data.dtype_size,  # Size of each element in bytes
                    tensor_fqn_data.shape_in_file,  # Full tensor shape
                    offsets_of_tensor_being_read,  # Where this shard belongs in the full tensor
                    metadata[SHAPE_KEY],  # Shape of this shard
                )

            output_stream.write(full_tensor_mv)


def _write_data(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
    num_threads: int = 1,
) -> None:
    """
    Write tensor data from input files to the output files using memory mapping.

    This function reads tensor data from each input file and writes it to the appropriate
    position in the output files based on the tensor's offsets. When num_threads > 1,
    the work is split across threads with each thread handling a different output file.

    Args:
        input_files_data: Dictionary mapping input file paths to their metadata
        output_files_data: Dictionary mapping output file paths to their metadata
        num_threads: Number of threads to use for parallel processing
    """
    if num_threads <= 1 or len(output_files_data) <= 1:
        # Sequential processing
        for output_file, output_data in output_files_data.items():
            _process_output_file(output_file, output_data, input_files_data)
    else:
        # Parallel processing with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(num_threads, len(output_files_data))
        ) as executor:
            futures = []
            for output_file, output_data in output_files_data.items():
                futures.append(
                    executor.submit(
                        _process_output_file,
                        output_file,
                        output_data,
                        input_files_data,
                    )
                )

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                # Handle any exceptions that might have occurred
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing output file: {e}")
                    raise


def _write_sub_tensor_to_file_optimized(
    full_tensor_mv: memoryview,
    sub_tensor_bytes: bytes,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
) -> None:
    """
    Optimized version that writes the maximum number of contiguous bytes possible.

    Uses a unified algorithm that calculates the maximum contiguous bytes that can be
    written in each iteration and continues until the entire subtensor is written.
    Handles all sharding patterns efficiently:
    - Full sub-tensor at once for row-wise sharding
    - Row-by-row for column-wise sharding
    - Optimized chunks for other patterns

    Args:
        full_tensor_mv: Buffer to write the full tensor to
        sub_tensor_bytes: Raw tensor data as bytes
        element_size: Size of each element in bytes
        tensor_shape: Shape of the full tensor
        sub_tensor_offsets: Starting offsets of the sub-tensor within the full tensor
        sub_tensor_shape: Shape of the sub-tensor
    """
    # Handle empty tensors
    if not tensor_shape or not sub_tensor_shape:
        return

    # Calculate tensor strides for efficient indexing
    tensor_strides = [1]
    for i in range(len(tensor_shape) - 1, 0, -1):
        tensor_strides.insert(0, tensor_strides[0] * tensor_shape[i])

    sub_tensor_strides = [1]
    for i in range(len(sub_tensor_shape) - 1, 0, -1):
        sub_tensor_strides.insert(0, sub_tensor_strides[0] * sub_tensor_shape[i])

    total_elements = math.prod(sub_tensor_shape)

    elements_written = 0
    while elements_written < total_elements:
        # Convert linear index to multi-dimensional indices
        temp_idx = elements_written
        indices = []
        for dim_size in reversed(sub_tensor_shape):
            indices.append(temp_idx % dim_size)
            temp_idx //= dim_size
        indices.reverse()

        # Calculate maximum contiguous elements we can write from this position
        max_contiguous = _calculate_max_contiguous_elements(
            indices, sub_tensor_shape, tensor_shape
        )

        # Calculate source position in bytes
        src_pos = sum(idx * stride for idx, stride in zip(indices, sub_tensor_strides))
        src_byte_offset = src_pos * element_size

        # Calculate destination position in bytes
        dest_indices = [
            idx + offset for idx, offset in zip(indices, sub_tensor_offsets)
        ]
        dest_pos = sum(
            idx * stride for idx, stride in zip(dest_indices, tensor_strides)
        )
        dest_byte_offset = dest_pos * element_size

        # Write the contiguous chunk
        bytes_to_write = max_contiguous * element_size
        chunk_data = sub_tensor_bytes[
            src_byte_offset : src_byte_offset + bytes_to_write
        ]
        full_tensor_mv[dest_byte_offset : dest_byte_offset + bytes_to_write] = (
            chunk_data
        )

        elements_written += max_contiguous


def _calculate_max_contiguous_elements(
    indices: list[int],
    sub_tensor_shape: list[int],
    tensor_shape: list[int],
) -> int:
    """
    Calculate the maximum number of contiguous elements that can be written from current position.

    This determines the largest chunk by checking how elements are laid out in memory
    and finding natural boundaries where contiguity breaks.

    Args:
        indices: Current position indices in the sub-tensor
        sub_tensor_shape: Shape of the sub-tensor being written
        tensor_shape: Shape of the full tensor

    Raises:
        ValueError: If input lists are empty, have mismatched lengths, or contain invalid values
    """
    # Validate input lists are not empty
    if not indices or not sub_tensor_shape or not tensor_shape:
        raise ValueError("Input lists cannot be empty")

    # Validate all lists have the same length (same number of dimensions)
    if not (len(indices) == len(sub_tensor_shape) == len(tensor_shape)):
        raise ValueError(
            f"All input lists must have the same length. Got indices: {len(indices)}, "
            f"sub_tensor_shape: {len(sub_tensor_shape)}, tensor_shape: {len(tensor_shape)}"
        )

    # Validate indices are within bounds of sub_tensor_shape
    for i, (idx, sub_dim) in enumerate(zip(indices, sub_tensor_shape)):
        if idx >= sub_dim:
            raise ValueError(
                f"Index {idx} at dimension {i} is out of bounds for sub-tensor shape {sub_tensor_shape}"
            )

    # Validate sub_tensor dimensions don't exceed tensor dimensions
    for i, (sub_dim, tensor_dim) in enumerate(zip(sub_tensor_shape, tensor_shape)):
        if sub_dim > tensor_dim:
            raise ValueError(
                f"Sub-tensor dimension {sub_dim} at position {i} exceeds tensor dimension {tensor_dim}"
            )

    # Start with elements remaining in the last dimension
    max_contiguous = sub_tensor_shape[-1] - indices[-1]

    # Check if we can extend across multiple dimensions
    # We can write across dimension boundaries if we're writing complete "rows"
    # and the layout in destination tensor maintains contiguity

    # For 2D case: check if we can write multiple complete rows
    if len(sub_tensor_shape) >= 2:
        # If we're at the start of a row and can write complete rows
        if indices[-1] == 0:  # At start of last dimension (column)
            rows_remaining = sub_tensor_shape[-2] - indices[-2]  # Rows left to write

            # Check if writing complete rows maintains contiguity in destination
            # This is true for row-wise sharding or when sub-tensor spans full width
            if sub_tensor_shape[-1] == tensor_shape[-1]:  # Full width
                max_contiguous = rows_remaining * sub_tensor_shape[-1]

            # For higher dimensions, check if we can extend further
            if len(sub_tensor_shape) >= 3 and indices[-2] == 0:
                # Check if we can write complete 2D slices
                remaining_in_dim = sub_tensor_shape[-3] - indices[-3]
                if (
                    sub_tensor_shape[-1] == tensor_shape[-1]
                    and sub_tensor_shape[-2] == tensor_shape[-2]
                ):
                    max_contiguous = (
                        remaining_in_dim * sub_tensor_shape[-2] * sub_tensor_shape[-1]
                    )

    return max_contiguous


def _write_overall_metadata_file(
    output_dir: str,
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Write the overall metadata file that maps tensor names to their file locations.

    This creates a model.safetensors.index.json file that HuggingFace models use
    to locate tensors across multiple files.

    Args:
        output_dir: Directory where the metadata file will be written
        output_files_data: Dictionary mapping output file paths to their metadata
    """
    total_size = 0
    weight_map = {}
    for output_path, value in output_files_data.items():
        for fqn, fqn_data in value.fqn_data.items():
            total_size += math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
            weight_map[fqn] = os.path.basename(output_path)

    metadata_to_write: dict[str, Any] = {}
    metadata_to_write["metadata"] = {"total_size": total_size}
    metadata_to_write["weight_map"] = weight_map

    metadata_path = os.path.join(output_dir, f"{_metadata_fn}")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_to_write, metadata_file, indent=2)


def _consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_file_mapping: dict[str, str],
    num_threads: int,
) -> dict[str, _OutputFileData]:
    output_files_data: dict[str, _OutputFileData] = {}
    # Create multiple output files based on the provided mapping
    for fqn, filename in fqn_to_file_mapping.items():
        output_path = os.path.join(output_dir, filename)

        if output_path not in output_files_data:
            output_files_data[output_path] = _OutputFileData(fqn_data={fqn: _FqnData()})
        else:
            output_files_data[output_path].fqn_data[fqn] = _FqnData()

    # Find all safetensors files in the input directory
    safetensors_files = glob.glob(os.path.join(input_dir, f"*{SUFFIX}"))

    # Read metadata from all input files
    input_files_data: dict[str, _InputFileData] = {}
    for safetensor_file in safetensors_files:
        with open(safetensor_file, "rb") as f:
            metadata, size = _get_safetensors_file_metadata(f)
            input_files_data[safetensor_file] = _InputFileData(
                metadata_size=size, metadata=metadata
            )
    # Step 1: Parse metadata to determine tensor shapes and types
    _parse_input_metadata(input_files_data, output_files_data)

    # Step 2: Write metadata headers to output files
    _write_metadata(output_files_data)
    # Step 3: Write actual tensor data from input files to output files
    _write_data(input_files_data, output_files_data, num_threads)

    return output_files_data


def consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
) -> None:
    """
    Main function to consolidate sharded safetensors files into one or more output files.

    This function orchestrates the entire consolidation process:
    1. Sets up the output file structure based on the fqn_to_index_mapping
    2. Finds all safetensors files in the input directory
    3. Parses metadata from all input files
    4. Writes metadata to the output files
    5. Writes tensor data from input files to output files
    6. Writes overall model.index.safetensors.json file with weight map

    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Optional mapping of tensor names to output file indices.
                             If None, all tensors will be consolidated into a single file.
        num_threads: Number of threads to use for parallel processing of saving data to output files.
    """
    start_time = time.time()
    logger.info(
        "Consolidating safetensors files from %s to %s. Beginning at time %f",
        input_dir,
        output_dir,
        start_time,
    )

    max_index = max(fqn_to_index_mapping.values())
    fqn_to_file_mapping = {
        fqn: _gen_file_name(idx, max_index) for fqn, idx in fqn_to_index_mapping.items()
    }

    output_files_data = _consolidate_safetensors_files(
        input_dir, output_dir, fqn_to_file_mapping, num_threads
    )

    # Step 4: Write overall model.index.safetensors.json file with weight map
    _write_overall_metadata_file(output_dir, output_files_data)

    logger.info("Done consolidating. Took %.2f secs.", time.time() - start_time)


def consolidate_safetensors_files_on_every_rank(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """
    Consolidate sharded safetensors files across multiple ranks, with each rank handling a subset of output files.

    This function distributes the consolidation work by assigning output files to different ranks.
    All tensors with the same index in fqn_to_index_mapping are processed by the same rank,
    as they belong to the same output file.

    If process_group is provided, rank and world_size will be derived from it. Otherwise,
    they will be automatically detected from the distributed environment if available.

    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Mapping of tensor names to output file indices
        num_threads: Number of threads to use for parallel processing on each rank
        process_group: PyTorch distributed process group (default: None, will use default group)
    """

    start_time = time.time()
    # Derive rank and world_size from process_group or default distributed environment
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
    else:
        # Default to single process mode if distributed is not initialized
        rank = 0
        world_size = 1
        logger.warning(
            "Distributed environment not initialized. Running in single process mode."
        )
    logger.info(
        "Rank %d/%d: Consolidating safetensors files from %s to %s",
        rank,
        world_size,
        input_dir,
        output_dir,
    )

    # Find all unique indices in the mapping
    unique_indices = set(fqn_to_index_mapping.values())

    # Distribute indices across ranks
    indices_for_this_rank = []
    for idx in unique_indices:
        # Simple distribution: index % world_size == rank
        if idx % world_size == rank:
            indices_for_this_rank.append(idx)

    logger.info(
        "Rank %d: Assigned %d output files out of %d total files",
        rank,
        len(indices_for_this_rank),
        len(unique_indices),
    )

    # Filter the fqn_to_index_mapping to only include tensors for this rank
    filtered_mapping = {
        fqn: idx
        for fqn, idx in fqn_to_index_mapping.items()
        if idx in indices_for_this_rank
    }

    output_files_data: dict[str, _OutputFileData] = {}
    if filtered_mapping:
        # Convert index mapping to filename mapping
        max_index = max(unique_indices)
        filtered_filename_mapping = {}
        for fqn, idx in filtered_mapping.items():
            filename = _gen_file_name(idx, max_index)
            filtered_filename_mapping[fqn] = filename

        # Call the existing consolidation function with the filtered mapping
        output_files_data = _consolidate_safetensors_files(
            input_dir=input_dir,
            output_dir=output_dir,
            fqn_to_file_mapping=filtered_filename_mapping,
            num_threads=num_threads,
        )

    logger.info(
        "Rank %d: Done consolidating. Processed %d unique indices in %.2f secs.",
        rank,
        len(indices_for_this_rank),
        time.time() - start_time,
    )

    # Wait for all ranks to complete and gather output_files_data on rank 0
    if dist.is_available() and dist.is_initialized():
        gathered_output_files_data: list[dict[str, _OutputFileData]] | None = (
            [{} for _ in range(world_size)] if rank == 0 else None
        )
        dist.gather_object(
            output_files_data,
            gathered_output_files_data,
            dst=0,
            group=process_group,
        )

        if rank == 0:
            # Merge all output_files_data from all ranks
            all_output_files_data: dict[str, _OutputFileData] = {}
            assert gathered_output_files_data is not None
            for rank_data in gathered_output_files_data:
                all_output_files_data.update(rank_data)

            _write_overall_metadata_file(output_dir, all_output_files_data)
            logger.info("Rank 0: Wrote overall metadata file.")
            logger.info("Total time taken: %.2f secs.", time.time() - start_time)
        dist.barrier(group=process_group)

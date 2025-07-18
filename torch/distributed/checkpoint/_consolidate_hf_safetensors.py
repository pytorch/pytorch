# pyre-strict

import concurrent.futures
import json
import logging
import math
import mmap
import os
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import fsspec  # type: ignore[import-untyped]
from fsspec.core import url_to_fs  # type: ignore[import-untyped]
from fsspec.implementations.local import LocalFileSystem  # type: ignore[import-untyped]

import torch
from torch.distributed.checkpoint._hf_utils import (
    _gen_file_name,
    _get_dcp_custom_metadata,
    _get_dtype,
    _get_safetensors_file_metadata,
    _metadata_fn,
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    FILE_NAME,
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
            # Add this tensor to the output file if it's already assigned there or if we're using a single output file
            if fqn in output_data.fqn_data or len(output_files_data) == 1:
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=tensor_size,
                    dtype_size=torch.finfo(_get_dtype(dtype_str)).bits
                    // 8,  # Convert bits to bytes
                    dtype_str=dtype_str,
                )


def _write_metadata(
    fs: fsspec.AbstractFileSystem,
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Write metadata to the beginning of each output safetensors file.

    This function writes the metadata section to each output file, including information
    about tensor shapes, data types, and offsets. It also updates the offset_in_file
    field for each tensor in the output_files_data.

    Args:
        fs: Filesystem interface for file operations
        output_files_data: Dictionary mapping output file paths to their metadata
    """
    # Process each output file
    for file_path, output_data in output_files_data.items():
        with fs.open(file_path, "wb") as f:
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
    input_fs: fsspec.AbstractFileSystem,
    file_path: str,
    start_offset: int,
    end_offset: int,
    metadata_size: int,
) -> bytes:
    """
    Read tensor data from a safetensors file using memory mapping for efficiency.

    Args:
        input_fs: Filesystem interface for input file operations
        file_path: Path to the safetensors file
        start_offset: Start offset of tensor data within the data section
        end_offset: End offset of tensor data within the data section
        metadata_size: Size of the metadata header

    Returns:
        Raw tensor data as bytes
    """
    # For local files, use mmap for efficient access
    if isinstance(input_fs, LocalFileSystem):
        # Local file - use mmap
        with open(file_path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                absolute_start = metadata_size + start_offset
                absolute_end = metadata_size + end_offset
                return bytes(mm[absolute_start:absolute_end])
    else:
        # Remote file - fall back to regular read
        with input_fs.open(file_path, "rb") as f:
            f.seek(metadata_size + start_offset)
            return f.read(end_offset - start_offset)


def _process_output_file(
    input_fs: fsspec.AbstractFileSystem,
    output_fs: fsspec.AbstractFileSystem,
    output_file: str,
    output_data: _OutputFileData,
    input_files_data: dict[str, _InputFileData],
) -> None:
    """
    Process a single output file by writing tensor data from input files using memory mapping.

    This function is designed to be run in parallel for different output files.

    Args:
        input_fs: Filesystem interface for input file operations
        output_fs: Filesystem interface for output file operations
        output_file: Path to the output file
        output_data: Metadata for the output file
        input_files_data: Dictionary mapping input file paths to their metadata
    """
    # Process each input safetensors file
    for safetensors_file in input_files_data.keys():
        file_metadata = input_files_data[safetensors_file].metadata
        input_metadata_size = input_files_data[safetensors_file].metadata_size

        for fqn, metadata in file_metadata.items():
            if fqn == DEFAULT_EXTRA_METADATA_KEY:
                continue

            # Skip if this tensor doesn't belong in this output file
            if fqn not in output_data.fqn_data:
                continue

            data_offsets = metadata[DATA_OFFSETS_KEY]

            # Use memory mapping to read tensor data efficiently
            data_to_write = _read_tensor_data_mmap(
                input_fs,
                safetensors_file,
                data_offsets[0],
                data_offsets[1],
                input_metadata_size,
            )

            # Get the offsets of this tensor shard within the full tensor
            custom_metadata = _get_dcp_custom_metadata(file_metadata)
            offsets_of_tensor_being_read = custom_metadata[fqn][SAVED_OFFSETS_KEY]  # type: ignore[index]

            # Get metadata for this tensor in the output file
            fqn_data = output_data.fqn_data[fqn]

            # Write this tensor shard to the appropriate position in the output file
            _write_sub_tensor_to_file_optimized(
                output_fs,
                data_to_write,
                fqn_data.dtype_size,  # Size of each element in bytes
                fqn_data.shape_in_file,  # Full tensor shape
                offsets_of_tensor_being_read,  # Where this shard belongs in the full tensor
                metadata[SHAPE_KEY],  # Shape of this shard
                output_file,
                # Calculate the exact byte position where this tensor data should start
                output_data.metadata_size + fqn_data.offset_in_file,
            )


def _write_data(
    input_fs: fsspec.AbstractFileSystem,
    output_fs: fsspec.AbstractFileSystem,
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
        input_fs: Filesystem interface for input file operations
        output_fs: Filesystem interface for output file operations
        input_files_data: Dictionary mapping input file paths to their metadata
        output_files_data: Dictionary mapping output file paths to their metadata
        num_threads: Number of threads to use for parallel processing
    """
    if num_threads <= 1 or len(output_files_data) <= 1:
        # Sequential processing
        for output_file, output_data in output_files_data.items():
            _process_output_file(
                input_fs, output_fs, output_file, output_data, input_files_data
            )
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
                        input_fs,
                        output_fs,
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
    fs: fsspec.AbstractFileSystem,
    sub_tensor_bytes: bytes,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
    output_file_path: str,
    output_start_byte: int,
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
        fs: Filesystem interface for file operations
        sub_tensor_bytes: Raw tensor data as bytes
        element_size: Size of each element in bytes
        tensor_shape: Shape of the full tensor
        sub_tensor_offsets: Starting offsets of the sub-tensor within the full tensor
        sub_tensor_shape: Shape of the sub-tensor
        output_file_path: Path to the output file
        output_start_byte: Starting byte position of the tensor in the file
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

    with fs.open(output_file_path, "r+b") as out_f:
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
            src_pos = sum(
                idx * stride for idx, stride in zip(indices, sub_tensor_strides)
            )
            src_byte_offset = src_pos * element_size

            # Calculate destination position in bytes
            dest_indices = [
                idx + offset for idx, offset in zip(indices, sub_tensor_offsets)
            ]
            dest_pos = sum(
                idx * stride for idx, stride in zip(dest_indices, tensor_strides)
            )
            dest_byte_offset = output_start_byte + dest_pos * element_size

            # Write the contiguous chunk
            bytes_to_write = max_contiguous * element_size
            out_f.seek(dest_byte_offset)
            chunk_data = sub_tensor_bytes[
                src_byte_offset : src_byte_offset + bytes_to_write
            ]
            out_f.write(chunk_data)

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
    """
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
    fs: fsspec.AbstractFileSystem,
    output_dir: str,
    output_files_data: dict[str, _OutputFileData],
) -> None:
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
    with fs.open(metadata_path, "w") as metadata_file:
        json.dump(metadata_to_write, metadata_file, indent=2)


def _upload_files_to_remote_fs(
    local_fs: fsspec.AbstractFileSystem,
    local_dir: str,
    output_fs: fsspec.AbstractFileSystem,
    output_dir: str,
) -> None:
    """
    Uploads the consolidated files to the remote filesystem.
    """
    for path in local_fs.ls(local_dir, detail=False):
        file = os.path.basename(path)
        model_str = FILE_NAME.split("-")[0]
        # Upload only the consolidated files with full tensors or the metadata file.
        # The check for file.startwith(model_str) is to ensure that we only upload
        # the consolidated files in the format "model-0000n-of-0000m.safetensors"
        # and not the files with sharded tensors.
        if file.endswith(SUFFIX) and file.startswith(model_str) or file == _metadata_fn:
            local_path = os.path.join(local_dir, file)
            remote_path = os.path.join(output_dir, file)
            output_fs.put_file(local_path, remote_path)


def consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: Optional[dict[str, int]] = None,
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
    # Create filesystem using fsspec for file operations
    input_fs, _ = url_to_fs(input_dir)
    output_fs, _ = url_to_fs(output_dir)

    if not isinstance(output_fs, LocalFileSystem):
        local_output_dir = tempfile.mkdtemp()
        logger.info("Created temporary directory %s", local_output_dir)
        local_output_fs, _ = url_to_fs(local_output_dir)
    else:
        local_output_fs = output_fs
        local_output_dir = output_dir

    # Initialize the output file structure
    output_files_data: dict[str, _OutputFileData] = {}
    if fqn_to_index_mapping is not None:
        # Create multiple output files based on the provided mapping
        for fqn, index in fqn_to_index_mapping.items():
            # Generate names like "model-00001-of-00005.safetensors"
            file_name = _gen_file_name(index, max(fqn_to_index_mapping.values()))
            output_path = os.path.join(local_output_dir, file_name)

            if output_path not in output_files_data:
                output_files_data[output_path] = _OutputFileData(
                    fqn_data={fqn: _FqnData()}
                )
            else:
                output_files_data[output_path].fqn_data[fqn] = _FqnData()
    else:
        # If no mapping is provided, create a single output file
        file_name = _gen_file_name(1, 1)
        output_path = os.path.join(local_output_dir, file_name)
        output_files_data[output_path] = _OutputFileData()

    # Find all safetensors files in the input directory
    safetensors_files = []
    for file in input_fs.ls(input_dir, detail=False):
        if file.endswith(SUFFIX):
            safetensors_files.append(file)

    # Read metadata from all input files
    input_files_data: dict[str, _InputFileData] = {}
    for safetensor_file in safetensors_files:
        with input_fs.open(safetensor_file, "rb") as f:
            metadata, size = _get_safetensors_file_metadata(f)
            input_files_data[safetensor_file] = _InputFileData(
                metadata_size=size, metadata=metadata
            )

    # Step 1: Parse metadata to determine tensor shapes and types
    _parse_input_metadata(input_files_data, output_files_data)

    # Step 2: Write metadata headers to output files
    _write_metadata(local_output_fs, output_files_data)

    # Step 3: Write actual tensor data from input files to output files
    _write_data(
        input_fs, local_output_fs, input_files_data, output_files_data, num_threads
    )

    # Step 4: Write overall model.index.safetensors.json file with weight map
    _write_overall_metadata_file(local_output_fs, local_output_dir, output_files_data)

    logger.info("Done consolidating. Took %.2f secs.", time.time() - start_time)

    if local_output_dir != output_dir:
        logger.info("Copying consolidated files to remote storage %s", output_dir)
        _upload_files_to_remote_fs(
            local_output_fs, local_output_dir, output_fs, output_dir
        )
        shutil.rmtree(local_output_dir)
        logger.info("Deleting temporary directory %s", local_output_dir)

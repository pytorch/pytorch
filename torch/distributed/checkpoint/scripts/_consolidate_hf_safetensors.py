# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
This script consolidates distributed checkpoint (DCP) HuggingFace safetensors files.

It takes sharded safetensors files created by DCP and combines them into one or more
consolidated files. This is useful for converting distributed model checkpoints into
a format that can be loaded by standard HuggingFace tools. The combination is done
through a simple metadata parsing and tensor data copying process.
"""

import argparse
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, List, Optional

import fsspec
import torch

from fsspec.core import url_to_fs
from safetensors import deserialize
from torch.distributed.checkpoint._hf_storage import (
    _gen_file_name,
    _get_dcp_custom_metadata,
    _get_dtype,
    _get_safetensors_file_metadata,
    DATA_KEY,
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
)


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
    

def _parse_input_metadata(safetensors_metadatas: List[Any], output_files_data: dict[str, _OutputFileData]) -> None:
    """
    Parse metadata from input safetensors files to determine the full tensor shapes and types.
    
    This function analyzes the metadata from all input files to determine the complete shape
    of each tensor after consolidation. It updates the output_files_data with this information.
    
    Args:
        safetensors_metadatas: List of metadata from input safetensors files
        output_files_data: Dictionary mapping output file paths to their metadata
    
    Raises:
        ValueError: If no DCP custom metadata is found in a safetensors file
    """
    # Dictionary to track the full size of each tensor across all shards
    fqn_to_size_mapping : dict[str, tuple[list[int], str]]= {}

    for safetensors_metadata in safetensors_metadatas:
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
        if not dcp_sharding_info:
            raise ValueError("No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated.")

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
        for _, output_data in output_files_data.items():
            # Add this tensor to the output file if it's already assigned there or if we're using a single output file
            if fqn in output_data.fqn_data or len(output_files_data) == 1:
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=tensor_size,
                    dtype_size=torch.finfo(_get_dtype(dtype_str)).bits // 8,  # Convert bits to bytes
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
                end_offset = curr_offset + math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
                
                # Store metadata for this tensor
                metadata[fqn] = {
                        SHAPE_KEY: fqn_data.shape_in_file,
                        DTYPE_KEY: fqn_data.dtype_str,
                        DATA_OFFSETS_KEY: [curr_offset, end_offset],  # Start and end byte offsets
                }
                # Store the offset for later use when writing the actual tensor data
                fqn_data.offset_in_file = curr_offset

                # Update current offset for the next tensor
                curr_offset = end_offset

            # Convert metadata to JSON and encode as bytes
            json_metadata = json.dumps(metadata)
            json_bytes = json_metadata.encode('utf-8')
            
            # Write the metadata size as an 8-byte unsigned integer (little-endian)
            size_in_bytes = len(json_bytes)
            header_len = struct.pack('<Q', size_in_bytes)

            # Write the header length and metadata to the file
            f.write(header_len)
            f.write(json_bytes)

            # Store the total metadata size (header + JSON) for later use
            output_data.metadata_size = f.tell()

def _write_data(
    input_fs: fsspec.AbstractFileSystem,
    output_fs: fsspec.AbstractFileSystem,
    input_safetensors_files: List[str],
    input_metadatas: dict[str, Any],
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Write tensor data from input files to the output files.
    
    This function reads tensor data from each input file and writes it to the appropriate
    position in the output files based on the tensor's offsets.
    
    Args:
        fs: Filesystem interface for file operations
        input_safetensors_files: List of input safetensors file paths
        input_metadatas: Dictionary mapping input file paths to their metadata
        output_files_data: Dictionary mapping output file paths to their metadata
    """
    # Process each input safetensors file
    for safetensors_file in input_safetensors_files:
        with input_fs.open(safetensors_file, "rb") as f:
            # Deserialize the safetensors file to get tensor data
            deserialized = deserialize(f.read())

            # Process each tensor in the file
            for fqn, val in deserialized:
                # Get the tensor data as bytes
                data_to_write = val[DATA_KEY]
                
                # Get the offsets of this tensor shard within the full tensor
                offsets_of_tensor_being_read = _get_dcp_custom_metadata(input_metadatas[safetensors_file])[fqn][SAVED_OFFSETS_KEY]

                # Find which output file(s) this tensor belongs to
                for output_file, output_data in output_files_data.items():
                    # Skip if this tensor doesn't belong in this output file
                    if fqn not in output_data.fqn_data:
                        continue

                    # Get metadata for this tensor in the output file
                    fqn_data = output_data.fqn_data[fqn]
                    
                    # Write this tensor shard to the appropriate position in the output file
                    _write_sub_tensor_to_file(
                        output_fs,
                        data_to_write,
                        fqn_data.dtype_size,  # Size of each element in bytes
                        fqn_data.shape_in_file,  # Full tensor shape
                        offsets_of_tensor_being_read,  # Where this shard belongs in the full tensor
                        val[SHAPE_KEY],  # Shape of this shard
                        output_file,
                        # Calculate the exact byte position where this tensor data should start
                        output_data.metadata_size + fqn_data.offset_in_file,
                    )
                    
def _write_row_wise_tensor(
    fs: fsspec.AbstractFileSystem,
    sub_tensor_bytes: bytearray,
    element_size: int,
    full_tensor_strides: list[int],
    sub_tensor_strides: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
    output_file_path: str,
    output_start_byte: int,
):
    """
    Writes a row-wise sharded tensor to the output file.
    
    This is an optimized path for tensors that are sharded along the first dimension,
    with all other dimensions being complete. This allows writing entire rows at once.
    
    Args:
        fs: Filesystem interface for file operations
        sub_tensor_bytes: Byte array containing the sub-tensor data
        element_size: The size of each element in bytes
        full_tensor_strides: Strides of the full tensor
        sub_tensor_strides: Strides of the sub-tensor
        sub_tensor_offsets: The starting offsets of the sub-tensor within the full tensor
        sub_tensor_shape: The shape of the sub-tensor
        output_file_path: The path to the file where the full tensor is stored
        output_start_byte: The starting byte of the full tensor in the file
    """
    # Open the output file in read+binary mode to allow seeking and writing
    with fs.open(output_file_path, "r+b") as out_f:
        # Calculate the number of elements in each row
        elements_per_row = full_tensor_strides[0]  # This is the stride of the first dimension
        
        # For each row in the sub-tensor
        for row_idx in range(sub_tensor_shape[0]):
            # Calculate the row index in the full tensor
            full_row_idx = sub_tensor_offsets[0] + row_idx
            
            # Calculate the position in the full tensor
            full_pos = full_row_idx * full_tensor_strides[0]
            full_byte_offset = output_start_byte + full_pos * element_size
            
            # Calculate the position in the sub-tensor
            sub_pos = row_idx * sub_tensor_strides[0]
            sub_byte_offset = sub_pos * element_size
            
            # Extract the row data from the sub-tensor
            row_size = elements_per_row * element_size
            row_data = sub_tensor_bytes[
                sub_byte_offset : sub_byte_offset + row_size
            ]
            
            # Seek to the correct position in the output file and write the data
            out_f.seek(full_byte_offset)
            out_f.write(row_data)

def _write_column_wise_tensor(
    fs: fsspec.AbstractFileSystem,
    sub_tensor_bytes: bytearray,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
    output_file_path: str,
    output_start_byte: int,
):
    """
    Writes a column-wise sharded 2D tensor to the output file.
    
    This is an optimized path for 2D tensors that are sharded along the second dimension,
    with the first dimension being complete. This requires writing column by column.
    
    Args:
        fs: Filesystem interface for file operations
        sub_tensor_bytes: Byte array containing the sub-tensor data
        element_size: The size of each element in bytes
        tensor_shape: The shape of the overall tensor
        sub_tensor_strides: Strides of the sub-tensor
        sub_tensor_offsets: The starting offsets of the sub-tensor within the full tensor
        sub_tensor_shape: The shape of the sub-tensor
        output_file_path: The path to the file where the full tensor is stored
        output_start_byte: The starting byte of the full tensor in the file
    """
    # Open the output file in read+binary mode to allow seeking and writing
    with fs.open(output_file_path, "r+b") as out_f:
        # For each column in the sub-tensor
        for col_idx in range(sub_tensor_shape[1]):
            # Calculate the column index in the full tensor
            full_col_idx = sub_tensor_offsets[1] + col_idx
            
            # For each row in the column
            for row_idx in range(sub_tensor_shape[0]):
                # Calculate the position in the full tensor
                full_pos = row_idx * tensor_shape[1] + full_col_idx
                full_byte_offset = output_start_byte + full_pos * element_size
                
                # Calculate the position in the sub-tensor
                sub_pos = row_idx * sub_tensor_shape[1] + col_idx
                sub_byte_offset = sub_pos * element_size
                
                # Extract the element data from the sub-tensor
                element_data = sub_tensor_bytes[
                    sub_byte_offset : sub_byte_offset + element_size
                ]
                
                # Seek to the correct position in the output file and write the data
                out_f.seek(full_byte_offset)
                out_f.write(element_data)

def _write_element_by_element(
    fs: fsspec.AbstractFileSystem,
    sub_tensor_bytes: bytearray,
    element_size: int,
    tensor_shape: list[int],
    full_tensor_strides: list[int],
    sub_tensor_strides: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
    output_file_path: str,
    output_start_byte: int,
):
    """
    Writes a sub-tensor to the output file using a general element-by-element approach.
    
    This is a general approach that works for any sharding pattern, but is less efficient
    than the specialized approaches for row-wise or column-wise sharding.
    
    Args:
        fs: Filesystem interface for file operations
        sub_tensor_bytes: Byte array containing the sub-tensor data
        element_size: The size of each element in bytes
        tensor_shape: The shape of the overall tensor
        full_tensor_strides: Strides of the full tensor
        sub_tensor_strides: Strides of the sub-tensor
        sub_tensor_offsets: The starting offsets of the sub-tensor within the full tensor
        sub_tensor_shape: The shape of the sub-tensor
        output_file_path: The path to the file where the full tensor is stored
        output_start_byte: The starting byte of the full tensor in the file
    """
    # Open the output file in read+binary mode to allow seeking and writing
    with fs.open(output_file_path, "r+b") as out_f:
        # Create a list to hold the current indices for each dimension
        indices = [0] * len(tensor_shape)
        
        # Calculate the total number of elements in the sub-tensor
        total_elements = 1
        for dim_size in sub_tensor_shape:
            total_elements *= dim_size
        
        # Process each element in the sub-tensor
        for element_idx in range(total_elements):
            # Calculate the indices for this element in the sub-tensor
            sub_idx = element_idx
            for dim in range(len(sub_tensor_shape) - 1, -1, -1):
                indices[dim] = sub_idx % sub_tensor_shape[dim]
                sub_idx //= sub_tensor_shape[dim]
            
            # Calculate the position of this element in the sub-tensor's byte array
            sub_pos = 0
            for dim in range(len(sub_tensor_shape)):
                sub_pos += indices[dim] * sub_tensor_strides[dim]
            sub_byte_offset = sub_pos * element_size
            
            # Calculate the position of this element in the full tensor
            full_pos = 0
            for dim in range(len(tensor_shape)):
                # The global index is the local index plus the offset for this dimension
                global_idx = indices[dim] + sub_tensor_offsets[dim]
                full_pos += global_idx * full_tensor_strides[dim]
            full_byte_offset = output_start_byte + full_pos * element_size
            
            # Extract the element data from the sub-tensor
            element_data = sub_tensor_bytes[
                sub_byte_offset : sub_byte_offset + element_size
            ]
            
            # Seek to the correct position in the output file and write the data
            out_f.seek(full_byte_offset)
            out_f.write(element_data)

def _write_sub_tensor_to_file(
    fs: fsspec.AbstractFileSystem,
    sub_tensor_bytes: bytearray,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
    output_file_path: str,
    output_start_byte: int,
):
    """
    Writes a sub-tensor from a byte array into a file representing the full tensor at specified offsets.

    This function handles the complex task of placing a tensor shard (sub-tensor) at the correct
    position within the consolidated tensor file. It works by calculating the exact byte offsets
    for each slice of data and writing them to the appropriate positions. This implementation
    supports tensors of any dimensionality with optimized paths for common sharding patterns:
    - Row-wise sharding (optimized path)
    - Column-wise sharding for 2D tensors (optimized path)
    - Any other arbitrary sharding pattern (general element-by-element approach)

    Args:
        sub_tensor_bytes: Byte array containing the sub-tensor data
        element_size: The size of each element in bytes
        tensor_shape: The shape of the overall tensor (list)
        sub_tensor_offsets: The starting offsets of the sub-tensor within the full tensor (list)
        sub_tensor_shape: The shape of the sub-tensor (list)
        output_file_path: The path to the file where the full tensor is stored
        output_start_byte: The starting byte of the full tensor in the file
    """
    # Handle the case of empty tensors
    if not tensor_shape or not sub_tensor_shape:
        return

    # Calculate strides for the full tensor (row-major order, C-style)
    # Stride is the number of elements to skip to move to the next element in that dimension
    full_tensor_strides = [1] * len(tensor_shape)
    for i in range(len(tensor_shape) - 2, -1, -1):
        full_tensor_strides[i] = full_tensor_strides[i + 1] * tensor_shape[i + 1]

    # Calculate strides for the sub-tensor (row-major order, C-style)
    sub_tensor_strides = [1] * len(sub_tensor_shape)
    for i in range(len(sub_tensor_shape) - 2, -1, -1):
        sub_tensor_strides[i] = sub_tensor_strides[i + 1] * sub_tensor_shape[i + 1]

    # Check if this is a row-wise sharded tensor
    # Row-wise sharding is detected when the last dimension is complete
    # and only the first dimension is partial
    is_row_wise = False
    if len(tensor_shape) >= 2:
        # Check if all dimensions except the first are complete
        all_other_dims_complete = True
        for i in range(1, len(tensor_shape)):
            if sub_tensor_shape[i] != tensor_shape[i]:
                all_other_dims_complete = False
                break
        
        # Row-wise sharding: first dimension is partial, all others are complete
        is_row_wise = all_other_dims_complete and sub_tensor_shape[0] < tensor_shape[0]
    
    # Check if this is a column-wise sharded 2D tensor
    # Column-wise sharding is detected when the first dimension is complete
    # and the second dimension is partial (only for 2D tensors)
    is_column_wise = False
    if len(tensor_shape) == 2:
        is_column_wise = (sub_tensor_shape[0] == tensor_shape[0] and 
                         sub_tensor_shape[1] < tensor_shape[1])
    
    # Call the appropriate function based on the sharding pattern
    if is_row_wise:
        _write_row_wise_tensor(
            fs,
            sub_tensor_bytes,
            element_size,
            full_tensor_strides,
            sub_tensor_strides,
            sub_tensor_offsets,
            sub_tensor_shape,
            output_file_path,
            output_start_byte,
        )
    elif is_column_wise:
        _write_column_wise_tensor(
            fs,
            sub_tensor_bytes,
            element_size,
            tensor_shape,
            sub_tensor_offsets,
            sub_tensor_shape,
            output_file_path,
            output_start_byte,
        )
    else:
        _write_element_by_element(
            fs,
            sub_tensor_bytes,
            element_size,
            tensor_shape,
            full_tensor_strides,
            sub_tensor_strides,
            sub_tensor_offsets,
            sub_tensor_shape,
            output_file_path,
            output_start_byte,
        )

def consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: Optional[dict[str, int]] = None,
) -> None:
    """
    Main function to consolidate sharded safetensors files into one or more output files.
    
    This function orchestrates the entire consolidation process:
    1. Sets up the output file structure based on the fqn_to_index_mapping
    2. Finds all safetensors files in the input directory
    3. Parses metadata from all input files
    4. Writes metadata to the output files
    5. Writes tensor data from input files to output files
    
    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Optional mapping of tensor names to output file indices.
                             If None, all tensors will be consolidated into a single file.
    """
    # Create filesystem using fsspec for file operations
    input_fs, _ = url_to_fs(input_dir)
    output_fs, _ = url_to_fs(output_dir)

    # Initialize the output file structure
    output_files_data : dict[str, _OutputFileData] = {}
    if fqn_to_index_mapping is None:
        # If no mapping is provided, create a single output file
        file_name = _gen_file_name(1, 1)  # Generate name like "model.safetensors"
        output_path = f"{output_dir}/{file_name}"
        output_files_data[output_path] = _OutputFileData()
    else:
        # Create multiple output files based on the provided mapping
        for fqn, index in fqn_to_index_mapping.items():
            # Generate names like "model-00001-of-00005.safetensors"
            file_name = _gen_file_name(index, max(fqn_to_index_mapping.values()))
            output_path = f"{output_dir}/{file_name}"

            # Create output file data structure if it doesn't exist yet
            if output_path not in output_files_data:
                output_files_data[output_path] = _OutputFileData(fqn_data={fqn: _FqnData()})
            else:
                output_files_data[output_path].fqn_data[fqn] = _FqnData()

    # Find all safetensors files in the input directory
    safetensors_files = []
    for file in input_fs.ls(input_dir, detail=False):
        if file.endswith(SUFFIX):
            safetensors_files.append(file)

    # Read metadata from all input files
    input_safetensors_metadatas = {}
    for safetensor_file in safetensors_files:
        with input_fs.open(safetensor_file, "rb") as f:
            input_safetensors_metadatas[safetensor_file] = _get_safetensors_file_metadata(f)
    
    # Step 1: Parse metadata to determine tensor shapes and types
    _parse_input_metadata(input_safetensors_metadatas.values(), output_files_data)

    # Step 2: Write metadata headers to output files
    _write_metadata(output_fs, output_files_data)

    # Step 3: Write actual tensor data from input files to output files
    _write_data(input_fs, output_fs, safetensors_files, input_safetensors_metadatas, output_files_data)

def main() -> None:
    """
    Command-line entry point for the consolidation script.
    
    Parses command-line arguments and calls consolidate_safetensors_files with the provided parameters.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Consolidate DCP sharded HuggingFace safetensors files")
    
    # Define required and optional arguments
    parser.add_argument(
        "input_path", 
        type=str, 
        required=True, 
        help="Path to directory containing sharded safetensors files"
    )
    parser.add_argument(
        "output_path", 
        type=str, 
        required=True, 
        help="Path to write consolidated safetensors files. Must be different from input path"
    )
    parser.add_argument(
        "fqn_to_index_mapping", 
        type=dict[str, int], 
        required=False,
        help="Mapping of which tensor names should belong to which consolidated file. If not provided, all tensors will be consolidated into one file. Expects numbers from 1 to N, where N is the number of files."
    )
    
    # Parse arguments and call the main function
    args = parser.parse_args()
    consolidate_safetensors_files(args.input_path, args.output_path, args.fqn_to_index_mapping)

if __name__ == "__main__":
    main()

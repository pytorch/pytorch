# mypy: allow-untyped-defs
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint._hf_utils import _metadata_fn
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadPlanner, ReadItem
from .hf_storage import HuggingFaceStorageReader


logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["QuantizedHuggingFaceStorageReader"]


class QuantizedHuggingFaceStorageReader(HuggingFaceStorageReader):
    """
    Extension of HuggingFaceStorageReader that handles quantized tensors.
    Checkpoint should have the full tensor in a SafeTensor file. The quantized
    tensor should not be sharded across multiple files.

    This reader handles the dequantization of tensors during the read process,
    converting them from quantized blocks to full dequantized tensors before
    copying to the target tensor.
    """

    def __init__(
        self,
        path: str,
        thread_count: int = 1,
        target_dtype: torch.dtype = torch.float32,
        block_size: int = 128,
    ):
        """
        Initialize the HuggingFace storage reader to load quantized checkpoints

        Args:
            path: directory where the checkpoint will be read from.
            thread_count: Number of threads to use to read distributed checkpoint. Defaults to 1.
            target_dtype: Target dtype for dequantized tensor. Defaults to torch.float32.
            block_size: Fixed block size for dequantization. Defaults to 128.
        """
        super().__init__(path=path, thread_count=thread_count)

        self.target_dtype: torch.dtype = target_dtype
        self.block_size: int = block_size
        self._weight_scale_mapping: dict[str, str] = {}
        # Track which file contains each tensor
        self._weight_map: dict[str, str] = {}
        # Cache for full tensor shapes (fqn -> shape)
        self._tensor_full_shapes: dict[str, torch.Size] = {}

    def read_metadata(self) -> Any:
        metadata = super().read_metadata()

        # Load quantization metadata first.
        self._load_quantization_metadata()

        # Build a cache of FQN -> full tensor shape, correcting for quantized tensors.
        for fqn, tensor_metadata in metadata.state_dict_metadata.items():
            # Only process TensorStorageMetadata which has size attribute.
            if isinstance(tensor_metadata, TensorStorageMetadata):
                # Check if this is a MXFP4 quantized tensor that needs shape correction.
                if fqn.endswith("_blocks"):
                    # Save the quantized tensor shapes for lookup when dequantization.
                    self._tensor_full_shapes[fqn + "_quantized"] = tensor_metadata.size
                    *prefix_shape, G, B = tensor_metadata.size
                    dequantized_size = torch.Size([*prefix_shape, G * B * 2])

                    # Update the metadata with the size after dequantization.
                    # Metadata used by planner to slice state dict.
                    tensor_metadata.size = dequantized_size
                    self._tensor_full_shapes[fqn] = dequantized_size
                else:
                    self._tensor_full_shapes[fqn] = tensor_metadata.size

        return metadata

    def _load_quantization_metadata(self):
        """Load quantization metadata from the checkpoint."""
        checkpoint_path = Path(self.path)
        # Load weight mapping from index file
        index_file = checkpoint_path / _metadata_fn

        with open(index_file) as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            self._build_weight_scale_mapping(weight_map)

    def _build_weight_scale_mapping(self, weight_map: dict[str, str]):
        """Analyze and build weight-scale tensor pairs from weight mapping."""
        # Store the complete weight map for file location lookups.
        self._weight_map = weight_map

        for tensor_name in weight_map:
            if tensor_name.endswith(".weight_scale_inv"):
                weight_name = tensor_name.replace(".weight_scale_inv", ".weight")
                if weight_name in weight_map:
                    self._weight_scale_mapping[weight_name] = tensor_name
            # Handle MXFP4 format: _blocks and _scales.
            elif tensor_name.endswith("_scales"):
                blocks_name = tensor_name.replace("_scales", "_blocks")
                if blocks_name in weight_map:
                    self._weight_scale_mapping[blocks_name] = tensor_name

    def _process_read_request(
        self, f: Any, req: ReadItem, planner: LoadPlanner
    ) -> None:
        """Override the Helper function that processes a single read request."""
        tensor_fqn = req.storage_index.fqn

        # Check if this is a quantized tensor that needs dequantization
        if self._is_tensor_quantized(tensor_fqn):
            tensor = self._read_quantized_tensor_with_block_alignment(req, f)
        else:
            # Standard tensor reading
            slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(req.storage_offsets, req.lengths)
            )
            tensor = f.get_slice(tensor_fqn)[slices]

        target_tensor = planner.resolve_tensor(req).detach()

        if target_tensor.size() != tensor.size():
            raise AssertionError(
                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
            )

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)

    def _get_slice_to_block_mapping(
        self, req: ReadItem
    ) -> tuple[tuple[int, int], tuple[int, int], slice, slice]:
        """
        Calculate which blocks correspond to the requested slice.

        Args:
            req: Read request containing tensor info and required slices

        Returns:
            Tuple of (row_block_range, col_block_range, row_slice, col_slice)
        """
        # Get the slice information
        row_slice = slice(
            req.storage_offsets[0], req.storage_offsets[0] + req.lengths[0]
        )
        col_slice = slice(
            req.storage_offsets[1], req.storage_offsets[1] + req.lengths[1]
        )

        # Calculate which blocks this slice spans
        row_start_block = row_slice.start // self.block_size
        row_end_block = (row_slice.stop - 1) // self.block_size + 1  # Inclusive end

        col_start_block = col_slice.start // self.block_size
        col_end_block = (col_slice.stop - 1) // self.block_size + 1  # Inclusive end

        return (
            (row_start_block, row_end_block),
            (col_start_block, col_end_block),
            row_slice,
            col_slice,
        )

    def _dequantize_tensor_mxfp4(
        self,
        blocks: torch.Tensor,
        scales: torch.Tensor,
        req: ReadItem,
        group_start: int,
        offset_in_first_group: int,
    ) -> torch.Tensor:
        """
        Dequantize a 4D tensor using MXFP4 format.
        Adapted from openai's implementation:
        https://github.com/openai/gpt-oss/blob/8890e95919f975a490fc0ba09ffb10890ec7319d/gpt_oss/torch/weights.py#L68

        Args:
            blocks: Sliced quantized weight tensor of shape [a_slice, b_slice, groups_slice, B] in uint8
            scales: FULL scale tensor of shape [a, b, c] in uint8 (will be converted to exponents)
            req: Read request containing slice information
            group_start: The starting group index in the checkpoint
            offset_in_first_group: Offset in values within the first group

        Returns:
            Dequantized tensor matching the requested shape
        """
        # FP4 lookup table
        FP4_VALUES = [
            +0.0,
            +0.5,
            +1.0,
            +1.5,
            +2.0,
            +3.0,
            +4.0,
            +6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]

        # blocks: [a_slice, b_slice, groups_slice, B] uint8.
        # Read slightly more groups than needed, and slice at the end.

        # Slice the scales to match the blocks dimensions.
        # [a_full, b_full, c_full] -> [a_slice, b_slice, groups_slice]
        dim0_start = req.storage_offsets[0]
        dim0_end = dim0_start + req.lengths[0]
        dim1_start = req.storage_offsets[1]
        dim1_end = dim1_start + req.lengths[1]
        num_groups = blocks.shape[2]
        scales = scales[
            dim0_start:dim0_end,
            dim1_start:dim1_end,
            group_start : group_start + num_groups,
        ]

        scales = scales.to(torch.int32) - 127

        assert blocks.shape[:-1] == scales.shape, (
            f"{blocks.shape=} does not match {scales.shape=}"
        )

        lut = torch.tensor(FP4_VALUES, dtype=self.target_dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(
            rows_total, B * 2, dtype=self.target_dtype, device=blocks.device
        )

        rows_per_chunk = 16384 * 512

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]

            # nibble indices -> int64
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)

            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]

            torch.ldexp(sub, exp, out=sub)

            del idx_lo, idx_hi, blk, exp

        result = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

        # Slice the last dimension to match the requested range.
        if offset_in_first_group > 0 or result.shape[-1] > req.lengths[2]:
            end_offset = offset_in_first_group + req.lengths[2]
            result = result[..., offset_in_first_group:end_offset]

        return result

    def _dequantize_tensor(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        full_tensor_shape: torch.Size,
        slice_info: tuple[tuple[int, int], tuple[int, int], slice, slice],
    ) -> torch.Tensor:
        """
        Dequantize a sliced tensor using the appropriate portion of the scale tensor.

        Args:
            weight: Sliced quantized weight tensor
            scale_inv: Full scale inverse tensor for dequantization
            full_tensor_shape: Shape of the original full tensor
            slice_info: Block mapping information from _get_slice_to_block_mapping

        Returns:
            Dequantized tensor
        """
        (row_block_range, col_block_range, row_slice, col_slice) = slice_info

        # Convert to float32 for computation
        # Certain quantized dtypes like Float8_e4m3fn
        # don't support multiplication on CPU yet in PyTorch.
        upcasted_weight = weight.to(torch.float32)

        # Create output tensor in target dtype
        dequantized = weight.detach().to(dtype=self.target_dtype, copy=True)

        # Get the actual slice boundaries
        row_start_global = row_slice.start
        row_end_global = row_slice.stop
        col_start_global = col_slice.start
        col_end_global = col_slice.stop

        # Apply scaling factors to each block that intersects with our slice
        for block_i in range(row_block_range[0], row_block_range[1]):
            for block_j in range(col_block_range[0], col_block_range[1]):
                # Calculate the block boundaries in global coordinates
                block_row_start_global = block_i * self.block_size
                block_row_end_global = min(
                    block_row_start_global + self.block_size, full_tensor_shape[0]
                )
                block_col_start_global = block_j * self.block_size
                block_col_end_global = min(
                    block_col_start_global + self.block_size, full_tensor_shape[1]
                )

                # Find the intersection of the block with our slice
                intersect_row_start = max(block_row_start_global, row_start_global)
                intersect_row_end = min(block_row_end_global, row_end_global)
                intersect_col_start = max(block_col_start_global, col_start_global)
                intersect_col_end = min(block_col_end_global, col_end_global)

                # Skip if no intersection
                if (
                    intersect_row_start >= intersect_row_end
                    or intersect_col_start >= intersect_col_end
                ):
                    continue

                # Convert global coordinates to local coordinates in the sliced tensor
                local_row_start = intersect_row_start - row_start_global
                local_row_end = intersect_row_end - row_start_global
                local_col_start = intersect_col_start - col_start_global
                local_col_end = intersect_col_end - col_start_global

                # Get the block from the sliced tensor
                block = upcasted_weight[
                    local_row_start:local_row_end, local_col_start:local_col_end
                ]

                # Apply the scale factor
                scale = scale_inv[block_i, block_j]
                block = block * scale

                # Convert block to target dtype and store
                block_converted = block.to(dtype=self.target_dtype)
                dequantized[
                    local_row_start:local_row_end, local_col_start:local_col_end
                ] = block_converted

        return dequantized

    def _is_tensor_quantized(self, tensor_fqn: str) -> bool:
        """
        Check if a tensor is a quantized.

        Args:
            tensor_fqn: Fully qualified name of the tensor

        Returns:
            True if tensor is quantized and has a corresponding scale tensor,
            False otherwise
        """
        # Skip scale tensors themselves
        if tensor_fqn.endswith((".weight_scale_inv", "_scales")):
            return False

        # Check if this weight tensor has a corresponding scale tensor
        if tensor_fqn not in self._weight_scale_mapping:
            return False

        return True

    def _read_quantized_tensor_with_block_alignment(
        self, req: ReadItem, safetensor_file: Any
    ) -> torch.Tensor:
        """
        Read a quantized tensor with block alignment.

        Args:
            req: Read request containing tensor info and required slices
            safetensor_file: Open safetensors file handle

        Returns:
            Dequantized tensor ready for use
        """
        tensor_fqn = req.storage_index.fqn
        scale_fqn = self._weight_scale_mapping[tensor_fqn]

        try:
            group_start = 0
            offset_in_first_group = 0
            if tensor_fqn.endswith("_blocks"):
                # Full tensor is a 4D MXFP4 quantized tensor: [..., G, B].
                # Each group G produces B * 2 dequantized values.
                # Checkpoint [..., G, B] -> dequantized [..., G*B*2].

                # The planner gives 3D requests based on the dequantized shape.
                # Need to figure out which groups (dimension 2 in checkpoint) to read.

                # Use the quantized checkpoint shape to get the correct B.
                *prefix_shape, B = self._tensor_full_shapes[tensor_fqn + "_quantized"]
                values_per_group = B * 2  # Each byte has 2 nibbles (4-bit values).

                # Calculate which groups we need based on the requested range in dim 2.
                # Ensure the reequest is in 3D.
                assert len(req.storage_offsets) == 3

                # Positions in dequantized space.
                dim2_start_deq = req.storage_offsets[2]
                dim2_length_deq = req.lengths[2]
                dim2_end_deq = dim2_start_deq + dim2_length_deq

                # Convert to group indices.
                group_start = dim2_start_deq // values_per_group
                group_end = (dim2_end_deq + values_per_group - 1) // values_per_group

                # Read only the necessary groups from checkpoint.
                weight_slices_4d = (
                    slice(
                        req.storage_offsets[0], req.storage_offsets[0] + req.lengths[0]
                    ),
                    slice(
                        req.storage_offsets[1], req.storage_offsets[1] + req.lengths[1]
                    ),
                    slice(group_start, group_end),
                    slice(None),  # Read all B values for each group.
                )
                quantized_tensor = safetensor_file.get_slice(tensor_fqn)[
                    weight_slices_4d
                ]

                # Also track the offset within the first group
                offset_in_first_group = dim2_start_deq - (
                    group_start * values_per_group
                )
            else:
                # 2D quantized tensor, use 2d block partition.
                weight_slices = tuple(
                    slice(offset, offset + length)
                    for offset, length in zip(req.storage_offsets, req.lengths)
                )
                quantized_tensor = safetensor_file.get_slice(tensor_fqn)[weight_slices]

            # Load the corresponding scale inverse tensor (full tensor)
            scale_file_name = self._weight_map.get(scale_fqn)
            if scale_file_name is None:
                raise ValueError(f"Scale tensor {scale_fqn} not found in weight_map")

            # Check if scale tensor is in the same file as the weight tensor
            weight_file_name = self._weight_map.get(tensor_fqn)

            if scale_file_name == weight_file_name:
                # Scale tensor is in the same file, use current handle
                scale_inv = safetensor_file.get_tensor(scale_fqn)
            else:
                # Scale tensor is in a different file, need to open it
                from safetensors import safe_open  # type: ignore[import]

                scale_file_path = Path(self.path) / scale_file_name
                with safe_open(
                    scale_file_path, framework="pt", device="cpu"
                ) as scale_file:
                    scale_inv = scale_file.get_tensor(scale_fqn)

            # Get the full tensor shape from our O(1) lookup cache
            full_tensor_shape = self._tensor_full_shapes.get(tensor_fqn)
            if full_tensor_shape is None:
                raise ValueError(f"Could not find full tensor shape for {tensor_fqn}")

            # Determine which dequantization function to use.
            if len(full_tensor_shape) == 2:
                # 2D block-wise quantization, e.g., used in deepseek v3.1
                slice_info = self._get_slice_to_block_mapping(req)
                dequantized_tensor = self._dequantize_tensor(
                    weight=quantized_tensor,
                    scale_inv=scale_inv,
                    full_tensor_shape=full_tensor_shape,
                    slice_info=slice_info,
                )
            elif tensor_fqn.endswith("_blocks"):
                # 4D with blocks along dimension 2, used in MXFP4, e.g. gpt-oss
                dequantized_tensor = self._dequantize_tensor_mxfp4(
                    blocks=quantized_tensor,
                    scales=scale_inv,
                    req=req,
                    group_start=group_start,
                    offset_in_first_group=offset_in_first_group,
                )
            else:
                raise ValueError("Unsupported quantization types")

            return dequantized_tensor

        except Exception as e:
            logger.error("Failed to read the quantized tensor!!")
            raise e

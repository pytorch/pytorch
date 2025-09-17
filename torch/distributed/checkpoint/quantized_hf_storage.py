# mypy: allow-untyped-defs
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint._hf_utils import _metadata_fn
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

    def read_metadata(self) -> Any:
        self._load_quantization_metadata()
        return super().read_metadata()

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
        # Store the complete weight map for file location lookups
        self._weight_map = weight_map

        for tensor_name in weight_map.keys():
            if tensor_name.endswith(".weight_scale_inv"):
                weight_name = tensor_name.replace(".weight_scale_inv", ".weight")
                if weight_name in weight_map:
                    self._weight_scale_mapping[weight_name] = tensor_name

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

        assert target_tensor.size() == tensor.size(), (
            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
        )

        target_tensor.copy_(tensor)
        planner.commit_tensor(req, target_tensor)

    def _calculate_scale_shape(
        self, weight: torch.Tensor, block_size: int
    ) -> tuple[int, int]:
        """Calculate expected scale tensor shape based on weight tensor and block size."""
        rows, cols = weight.shape
        block_rows = (rows + block_size - 1) // block_size  # Ceiling division
        block_cols = (cols + block_size - 1) // block_size  # Ceiling division
        return (block_rows, block_cols)

    def _dequantize_tensor(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize tensor using block-wise scaling.

        Args:
            weight: Quantized weight tensor
            scale_inv: Scale inverse tensor for dequantization

        Returns:
            Dequantized tensor
        """
        # Convert to float32 for computation
        # Certain quantized dtypes like Float8_e4m3fn
        # don't support multiplication on CPU yet in PyTorch.
        upcasted_weight = weight.to(torch.float32)

        # Get original dimensions
        orig_shape = weight.shape

        # Calculate block dimensions for the local shard
        expected_scale_shape = self._calculate_scale_shape(weight, self.block_size)
        block_rows, block_cols = expected_scale_shape

        # Create output tensor in target dtype
        dequantized = weight.detach().to(dtype=self.target_dtype, copy=True)

        # Apply scaling factors to each block
        for i in range(block_rows):
            row_start = i * self.block_size
            row_end = min(row_start + self.block_size, orig_shape[0])

            for j in range(block_cols):
                col_start = j * self.block_size
                col_end = min(col_start + self.block_size, orig_shape[1])

                # Get the block
                block = upcasted_weight[row_start:row_end, col_start:col_end]

                scale = scale_inv[i, j]
                block = block * scale

                # Explicitly convert block to target dtype
                block_converted = block.to(dtype=self.target_dtype)
                # Store the dequantized block
                dequantized[row_start:row_end, col_start:col_end] = block_converted

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
        if tensor_fqn.endswith(".weight_scale_inv"):
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
            # Load the quantized weight
            weight_slices = tuple(
                slice(offset, offset + length)
                for offset, length in zip(req.storage_offsets, req.lengths)
            )
            quantized_tensor = safetensor_file.get_slice(tensor_fqn)[weight_slices]

            # Load the corresponding scale inverse tensor
            # Use weight_map to find the correct file for the scale tensor
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

            # Perform dequantization
            dequantized_tensor = self._dequantize_tensor(
                weight=quantized_tensor,
                scale_inv=scale_inv,
            )

            return dequantized_tensor

        except Exception as e:
            logger.error("Failed to read the quantized tensor!!")
            raise e

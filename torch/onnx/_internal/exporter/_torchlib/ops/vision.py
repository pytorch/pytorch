"""ONNX implementations for torchvision operators."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TC001,TC002

from __future__ import annotations

import importlib

from onnxscript.onnx_opset import opset18 as op  # type: ignore[attr-defined]

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


# Only register the torchvision op if torchvision is installed, since
# torch.ops.torchvision is not available otherwise.
if importlib.util.find_spec("torchvision") is not None:
    import torchvision  # noqa: F401  # Loads the torchvision custom ops

    @onnx_impl(torch.ops.torchvision.roi_align.default, trace_only=True)
    def torchvision_roi_align(
        input: TReal,
        rois: TReal,
        spatial_scale: float,
        output_height: int,
        output_width: int,
        sampling_ratio: int,
        aligned: bool,
    ) -> TReal:
        """roi_align(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio, bool aligned) -> Tensor

        Torchvision passes rois as [num_rois, 5] where column 0 is the batch
        index, but ONNX RoiAlign expects rois as [num_rois, 4] with a separate
        batch_indices tensor.
        """
        # Split the 5-column rois tensor into batch_indices (col 0) and
        # roi coordinates (cols 1-4)
        batch_indices = op.Cast(
            op.Squeeze(op.Slice(rois, [0], [1], [1]), axes=[1]),
            to=7,  # INT64
        )
        roi_coords = op.Slice(rois, [1], [5], [1])

        # Negative sampling_ratio in torchvision means adaptive (let the op
        # decide). ONNX represents this as 0.
        if sampling_ratio < 0:
            sampling_ratio = 0

        coordinate_transformation_mode = (
            "half_pixel" if aligned else "output_half_pixel"
        )

        return op.RoiAlign(
            input,
            roi_coords,
            batch_indices,
            coordinate_transformation_mode=coordinate_transformation_mode,
            output_height=output_height,
            output_width=output_width,
            sampling_ratio=sampling_ratio,
            spatial_scale=spatial_scale,
        )

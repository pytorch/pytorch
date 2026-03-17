"""torchvision operators for ONNX export."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# pyrefly: ignore-errors
# ruff: noqa: TCH001,TCH002

from __future__ import annotations

import importlib.util
import logging

from onnxscript.onnx_opset import opset18 as op

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import INT64, TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


logger = logging.getLogger(__name__)


if importlib.util.find_spec("torchvision") is not None:
    try:
        import torchvision  # noqa: F401 -- triggers torchvision op registration

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
            """torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int output_height, int output_width, int sampling_ratio, bool aligned) -> Tensor"""

            # The torchvision rois tensor has shape [num_rois, 5] where column 0
            # is the batch index. ONNX RoiAlign expects batch_indices as a
            # separate 1-D int64 input and rois as [num_rois, 4].
            batch_indices = op.Cast(
                op.Squeeze(
                    op.Slice(rois, starts=[0], ends=[1], axes=[1]),
                    axes=[1],
                ),
                to=INT64.dtype,
            )
            rois_coords = op.Slice(rois, starts=[1], ends=[5], axes=[1])

            # torchvision aligned=True  -> ONNX "half_pixel"
            # torchvision aligned=False -> ONNX "output_half_pixel"
            coordinate_transformation_mode = (
                "half_pixel" if aligned else "output_half_pixel"
            )

            # ONNX sampling_ratio=0 means adaptive; torchvision uses negative values
            if sampling_ratio < 0:
                sampling_ratio = 0

            return op.RoiAlign(
                input,
                rois_coords,
                batch_indices,
                coordinate_transformation_mode=coordinate_transformation_mode,
                spatial_scale=spatial_scale,
                output_height=output_height,
                output_width=output_width,
                sampling_ratio=sampling_ratio,
            )

    except Exception:
        logger.debug("Failed to register torchvision ONNX ops", exc_info=True)

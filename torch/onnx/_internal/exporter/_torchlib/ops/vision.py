# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR vision translations."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import onnx_ir as ir

from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.onnx._internal.exporter._ir_builder import TranslationContext


_INT64_MAX = 9223372036854775807


@onnx_impl("torchvision::nms", opset_introduced=18)
def torchvision_nms(
    op: TranslationContext, boxes: ir.Value, scores: ir.Value, iou_threshold: float
) -> ir.Value:
    """nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor"""
    # boxes: [num_batches, spatial_dimension, 4]
    boxes = op.Unsqueeze(boxes, [0])
    # scores: [num_batches, num_classes, spatial_dimension]
    scores = op.Unsqueeze(scores, [0, 1])
    # nms_out: [num_selected_indices, 3] where each column is [batch_index, class_index, box_index]
    nms_out = op.NonMaxSuppression(boxes, scores, _INT64_MAX, iou_threshold)
    return op.Reshape(op.Slice(nms_out, [2], [3], [1]), [-1])


def _process_batch_indices_for_roi_align(op: TranslationContext, rois):
    # Extract batch indices from the first column (index 0) of rois
    indices = op.Slice(rois, [0], [1], [1])
    indices = op.Squeeze(indices, [1])
    return op.Cast(indices, to=ir.DataType.INT64)


def _process_rois_for_roi_align(op: TranslationContext, rois):
    # Extract roi coordinates from columns 1, 2, 3, 4 (x1, y1, x2, y2)
    return op.Slice(rois, [1], [5], [1])


def _process_sampling_ratio_for_roi_align(op: TranslationContext, sampling_ratio: int):
    if sampling_ratio < 0:
        warnings.warn(
            "ONNX export for RoIAlign with a non-zero sampling_ratio is not supported. "
            "The model will be exported with a sampling_ratio of 0.",
            stacklevel=2,
        )
        sampling_ratio = 0
    return sampling_ratio


@onnx_impl("torchvision::roi_align", opset_introduced=18)
def torchvision_roi_align(
    op: TranslationContext,
    input,
    rois,
    spatial_scale: float,
    pooled_height: int,
    pooled_width: int,
    sampling_ratio: int = -1,
    aligned: bool = False,
):
    """torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height, SymInt pooled_width, int sampling_ratio, bool aligned) -> Tensor"""
    batch_indices = _process_batch_indices_for_roi_align(op, rois)
    rois_coords = _process_rois_for_roi_align(op, rois)
    coordinate_transformation_mode = "half_pixel" if aligned else "output_half_pixel"
    sampling_ratio = _process_sampling_ratio_for_roi_align(op, sampling_ratio)

    return op.RoiAlign(
        input,
        rois_coords,
        batch_indices,
        coordinate_transformation_mode=coordinate_transformation_mode,
        spatial_scale=spatial_scale,
        output_height=pooled_height,
        output_width=pooled_width,
        sampling_ratio=sampling_ratio,
    )


@onnx_impl("torchvision::roi_pool", opset_introduced=18)
def torchvision_roi_pool(
    op: TranslationContext,
    input,
    boxes,
    output_size: Sequence[int],
    spatial_scale: float = 1.0,
):
    """roi_pool(input: torch.Tensor, boxes: Union[torch.Tensor, list[torch.Tensor]], output_size: None, spatial_scale: float = 1.0) -> torch.Tensor"""
    # MaxRoiPool expects boxes in format [batch_index, x1, y1, x2, y2]
    pooled_height, pooled_width = output_size
    return op.MaxRoiPool(
        input,
        boxes,
        pooled_shape=(pooled_height, pooled_width),
        spatial_scale=spatial_scale,
    )


@onnx_impl("torchvision::deform_conv2d", opset_introduced=19)
def torchvision_deform_conv2d(
    op: TranslationContext,
    input: ir.Value,
    weight: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None,
    bias: ir.Value | None,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
    offset_groups: int,
    use_mask: bool,
):
    """torchvision::deform_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor? mask, Tensor? bias, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups, int offset_groups, bool use_mask) -> Tensor"""

    return op.DeformConv(
        input,
        weight,
        offset,
        bias,
        mask if use_mask else None,
        dilations=(dilation_h, dilation_w),
        strides=(stride_h, stride_w),
        pads=(pad_h, pad_w, pad_h, pad_w),
        group=groups,
        offset_group=offset_groups,
    )

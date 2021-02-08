// Copyright 2004-present Facebook. All Rights Reserved.

#include <ATen/native/vulkan/ops/generate_proposals_op_util_boxes.h>
#include <torch/csrc/api/include/torch/types.h>

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor> BBoxTransformCPUKernel(
    const at::Tensor& roi_in,
    const at::Tensor& delta_in,
    const at::Tensor& iminfo_in,
    caffe2::vector<double> weights_,
    bool apply_scale_,
    bool rotated_,
    bool angle_bound_on_,
    int64_t angle_bound_lo_,
    int64_t angle_bound_hi_,
    double clip_angle_thresh_,
    bool legacy_plus_one_,
    c10::optional<std::vector<at::Tensor>> /* unused */ = {}
);

} // namespace fb
} // namespace caffe2

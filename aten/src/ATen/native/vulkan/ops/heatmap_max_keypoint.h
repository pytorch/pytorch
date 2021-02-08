// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/utils/eigen_utils.h"
#include <torch/csrc/api/include/torch/types.h>

namespace at {
namespace native {

torch::Tensor HeatmapMaxKeypointCPUKernel(
    const torch::Tensor& heatmaps_in,
    const torch::Tensor& bboxes_in,
    bool should_output_softmax_,
    c10::optional<std::vector<torch::Tensor>> /* unused */ = {}
);

} // namespace fb
} // namespace caffe2

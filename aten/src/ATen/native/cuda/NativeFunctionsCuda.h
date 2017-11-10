#pragma once

#include "ATen/ATen.h"
#include <tuple>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> SpatialRoIPooling_forward_cuda(
  const Tensor& input,
  const Tensor& rois,
  int64_t pooledHeight,
  int64_t pooledWidth,
  double spatialScale);

} // at::native
} // at

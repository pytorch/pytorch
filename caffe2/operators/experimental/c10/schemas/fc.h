#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct FullyConnected final {
  static constexpr const char* name = "FC";

  struct Cache final {
    vector<int64_t> Y_shape_cache_;
    Tensor bias_multiplier_ = Tensor{CPU};
  };

  using Signature = void(
      const Tensor& X,
      const Tensor& W,
      const Tensor& b,
      Tensor* output,
      int axis,
      int axis_w,
      Cache* cache,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 8> parameter_names = {
      {"X", "W", "b", "output", "axis", "axis_w", "cache", "context"}};
};

} // namespace ops
} // namespace caffe2

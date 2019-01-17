#pragma once

#include <c10/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace ops {

struct FullyConnected final {
  static constexpr const char* name = "FC";

  struct Cache final {
    vector<int64_t> Y_shape_cache_;
    C10Tensor bias_multiplier_ = C10Tensor(Tensor());
  };

  using Signature = void(
      const C10Tensor& X,
      const C10Tensor& W,
      const C10Tensor& b,
      const C10Tensor& output,
      int axis,
      int axis_w,
      Cache* cache);

  static constexpr size_t num_dispatch_args() {return 3;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"X", "W", "b", "output", "axis", "axis_w", "cache"}};
};

} // namespace ops
} // namespace caffe2

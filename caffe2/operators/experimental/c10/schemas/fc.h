#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/tensor.h"
#include <ATen/core/blob.h>

namespace caffe2 {
namespace ops {

struct FullyConnected final {
  static constexpr const char* name = "FC";

  struct State final {
    vector<int64_t> Y_shape_cache_;
    at::Tensor bias_multiplier_ = at::Tensor(C10Tensor(Tensor()));
  };

  using Signature = void(
      const at::Tensor& X,
      const at::Tensor& W,
      const at::Tensor& b,
      const at::Tensor& output,
      int axis,
      int axis_w,
      intrusive_ptr<Blob> state);

  static constexpr size_t num_dispatch_args() {return 3;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"X", "W", "b", "output", "axis", "axis_w", "cache"}};
};

} // namespace ops
} // namespace caffe2

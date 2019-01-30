#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/tensor.h"
#include <ATen/core/blob.h>
#include <ATen/core/dispatch/OpSchema.h>

namespace caffe2 {
namespace ops {

struct FullyConnected final {
  static constexpr const char* name = "FC";

  using Signature = void(
      const at::Tensor& X,
      const at::Tensor& W,
      const at::Tensor& b,
      const at::Tensor& output,
      int axis,
      int axis_w);

  static constexpr size_t num_dispatch_args() {return 3;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"X", "W", "b", "output", "axis", "axis_w"}};
};

} // namespace ops
} // namespace caffe2

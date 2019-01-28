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
      int64_t axis,
      int64_t axis_w);

  static constexpr size_t num_output_parameters() {return 1;}

  static constexpr c10::guts::array<const char*, 6> parameter_names() {
    return {"X", "W", "b", "output", "axis", "axis_w"};
  }
};

} // namespace ops
} // namespace caffe2

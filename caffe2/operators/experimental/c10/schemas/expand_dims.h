#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"
#include <ATen/core/ivalue.h>
#include <ATen/core/blob.h>
#include <ATen/core/dispatch/OpSchema.h>

namespace caffe2 {
namespace ops {

struct ExpandDims final {
  static constexpr const char* name = "expand_dims";

  using Signature = void(
      const at::Tensor& input,
      const at::Tensor& output,
      ArrayRef<int64_t> dims);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 3> parameter_names = {
      {"input", "output", "dims"}};
};

} // namespace ops
} // namespace caffe2

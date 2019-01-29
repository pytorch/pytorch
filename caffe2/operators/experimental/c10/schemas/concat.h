#pragma once

#include <ATen/core/dispatch/OpSchema.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include <c10/util/ArrayRef.h>
#include "caffe2/core/context_base.h"
#include <ATen/core/ivalue.h>

namespace caffe2 {
namespace ops {

struct Concat final {
  static constexpr const char* name = "concat";

  using Signature = void(
      ArrayRef<at::Tensor> inputs,
      const at::Tensor& output,
      const at::Tensor& split_info,
      int64_t add,
      int64_t add_axis);

  static constexpr size_t num_outputs() {return 2;}

  static constexpr c10::guts::array<const char*, 5> parameter_names() {
    return {"inputs", "output", "split_info_output", "add", "add_axis"};
  }
};

} // namespace ops
} // namespace caffe2

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"

namespace caffe2 {
namespace ops {

struct StopGradient final {
  static constexpr const char* name = "stop_gradient";

  using Signature = void(
      const at::Tensor& input,
      const at::Tensor& output);

  static constexpr size_t num_output_parameters() {return 1;}

  static constexpr c10::guts::array<const char*, 2> parameter_names() {
    return {"input", "output"};
  }
};

} // namespace ops
} // namespace caffe2

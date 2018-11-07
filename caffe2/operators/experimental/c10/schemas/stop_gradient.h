#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct StopGradient final {
  static constexpr const char* name = "stop_gradient";

  using Signature = void(
      const Tensor& input,
      Tensor* output,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 3> parameter_names = {
      {"input", "output", "context"}};
};

} // namespace ops
} // namespace caffe2

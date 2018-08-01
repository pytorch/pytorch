#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct StopGradient final {
  static constexpr const char* name = "stop_gradient";

  using Signature = void(
      const Tensor<CPUContext>& input,
      Tensor<CPUContext>* output,
      CPUContext* context);

  static constexpr c10::guts::array<const char*, 3> parameter_names = {
      {"input", "output", "context"}};
};

} // namespace ops
} // namespace caffe2

#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct Mul final {
  static constexpr const char* name = "mul";

  using Signature = void(
      const Tensor<CPUContext>& input1,
      const Tensor<CPUContext>& input2,
      Tensor<CPUContext>* output,
      bool legacy_broadcast,
      int axis,
      CPUContext* context);

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"input1", "input2", "output", "legacy_broadcast", "axis", "context"}};
};

} // namespace ops
} // namespace caffe2

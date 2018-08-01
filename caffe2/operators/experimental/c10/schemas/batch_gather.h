#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct BatchGather final {
  static constexpr const char* name = "batch_gather";

  using Signature = void(
      const Tensor<CPUContext>& data,
      const Tensor<CPUContext>& indices,
      Tensor<CPUContext>* output,
      CPUContext* context);

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"data", "indices", "output", "context"}};
};

} // namespace ops
} // namespace caffe2

#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct BatchGather final {
  static constexpr const char* name = "batch_gather";

  using Signature = void(
      const Tensor& data,
      const Tensor& indices,
      Tensor* output,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"data", "indices", "output", "context"}};
};

} // namespace ops
} // namespace caffe2

#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct Cast final {
  static constexpr const char* name = "cast";

  using Signature = void(
      const Tensor& input1,
      Tensor* output,
      TensorProto_DataType to);

  static constexpr c10::guts::array<const char*, 3> parameter_names = {
      {"input", "output", "to"}};
};

} // namespace ops
} // namespace caffe2

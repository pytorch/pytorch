#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct Flatten final {
  static constexpr const char* name = "flatten";

  using Signature = void(
      const Tensor& input,
      Tensor* output,
      int axis,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"input", "output", "axis", "context"}};
};

} // namespace ops
} // namespace caffe2

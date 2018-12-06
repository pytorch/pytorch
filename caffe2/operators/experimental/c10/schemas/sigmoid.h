#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct Sigmoid final {
  static constexpr const char* name = "sigmoid";

  using Signature =
      void(const Tensor& input, Tensor* output);

  static constexpr c10::guts::array<const char*, 2> parameter_names = {
      {"input", "output"}};
};

} // namespace ops
} // namespace caffe2

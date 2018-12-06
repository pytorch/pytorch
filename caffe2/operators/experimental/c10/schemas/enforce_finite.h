#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct EnforceFinite final {
  static constexpr const char* name = "enforce_finite";

  using Signature = void(const Tensor& input);

  static constexpr c10::guts::array<const char*, 1> parameter_names = {
      {"input"}};
};

} // namespace ops
} // namespace caffe2

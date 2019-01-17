#pragma once

#include <c10/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"

namespace caffe2 {
namespace ops {

struct Flatten final {
  static constexpr const char* name = "flatten";

  using Signature = void(
      const C10Tensor& input,
      const C10Tensor& output,
      int axis);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 3> parameter_names = {
      {"input", "output", "axis"}};
};

} // namespace ops
} // namespace caffe2

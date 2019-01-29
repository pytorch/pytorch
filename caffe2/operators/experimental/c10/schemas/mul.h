#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"

namespace caffe2 {
namespace ops {

struct Mul final {
  static constexpr const char* name = "mul";

  using Signature = void(
      const at::Tensor& input1,
      const at::Tensor& input2,
      const at::Tensor& output,
      bool legacy_broadcast,
      int axis);

  static constexpr size_t num_dispatch_args() {return 2;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 5> parameter_names = {
      {"input1", "input2", "output", "legacy_broadcast", "axis"}};
};

} // namespace ops
} // namespace caffe2

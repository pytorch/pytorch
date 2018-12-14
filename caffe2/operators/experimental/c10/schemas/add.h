#pragma once

#include <c10/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"

namespace caffe2 {
namespace ops {

struct Add final {
  static constexpr const char* name = "add";

  using Signature = void(
      const C10Tensor& input1,
      const C10Tensor& input2,
      const C10Tensor& output,
      bool legacy_broadcast,
      int axis,
      BaseContext* context);

  static constexpr size_t num_dispatch_args() {return 2;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"input1", "input2", "output", "legacy_broadcast", "axis", "context"}};
};

} // namespace ops
} // namespace caffe2

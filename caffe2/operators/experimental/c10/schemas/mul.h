#pragma once

/*
 * This op is only for testing the c10 dispatcher and might not support all
 * parameter combinations or backends the corresponding caffe2 op supports.
 * Please ignore this.
 * TODO Remove this comment once this is more final
 */

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct Mul final {
  static constexpr const char* name = "mul";

  using Signature = void(
      const Tensor& input1,
      const Tensor& input2,
      Tensor* output,
      bool legacy_broadcast,
      int axis,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"input1", "input2", "output", "legacy_broadcast", "axis", "context"}};
};

} // namespace ops
} // namespace caffe2

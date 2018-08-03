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

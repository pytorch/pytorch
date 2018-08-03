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

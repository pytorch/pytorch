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

struct BatchGather final {
  static constexpr const char* name = "batch_gather";

  using Signature = void(
      const Tensor& data,
      const Tensor& indices,
      Tensor* output,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"data", "indices", "output", "context"}};
};

} // namespace ops
} // namespace caffe2

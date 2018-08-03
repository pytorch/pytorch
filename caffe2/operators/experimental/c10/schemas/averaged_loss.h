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

struct AveragedLoss final {
  struct State final {
    Tensor scratch = Tensor{CPU};
  };

  static constexpr const char* name = "averaged_loss";

  using Signature = void(
      const Tensor& input,
      Tensor* output,
      State* state,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"input", "output", "state", "context"}};
};

} // namespace ops
} // namespace caffe2

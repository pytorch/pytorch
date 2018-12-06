#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

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

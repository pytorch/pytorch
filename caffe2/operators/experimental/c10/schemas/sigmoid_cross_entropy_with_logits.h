#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct SigmoidCrossEntropyWithLogits final {
  static constexpr const char* name = "sigmoid_cross_entropy_with_logits";

  using Signature = void(
      const Tensor& input1,
      const Tensor& input2,
      Tensor* output,
      bool log_D_trick,
      bool unjoined_lr_loss);

  static constexpr c10::guts::array<const char*, 5> parameter_names = {
      {"input1", "input2", "output", "log_d_trick", "unjoined_lr_loss"}};
};

} // namespace ops
} // namespace caffe2

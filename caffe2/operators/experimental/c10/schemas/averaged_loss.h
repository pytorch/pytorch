#pragma once

#include <c10/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace ops {

struct AveragedLoss final {
  struct State final {
    C10Tensor scratch = C10Tensor(empty({}, CPU));
  };

  static constexpr const char* name = "averaged_loss";

  using Signature = void(
      const C10Tensor& input,
      const C10Tensor& output,
      State* state,
      BaseContext* context);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 4> parameter_names = {
      {"input", "output", "state", "context"}};
};

} // namespace ops
} // namespace caffe2

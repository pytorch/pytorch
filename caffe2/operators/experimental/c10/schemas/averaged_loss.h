#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"
#include "caffe2/core/tensor.h"
#include <ATen/core/blob.h>

namespace caffe2 {
namespace ops {

struct AveragedLoss final {
  struct State final {
    at::Tensor scratch = at::Tensor(C10Tensor(empty({}, CPU)));
  };

  static constexpr const char* name = "averaged_loss";

  using Signature = void(
      const at::Tensor& input,
      const at::Tensor& output,
      intrusive_ptr<Blob> state);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 3> parameter_names = {
      {"input", "output", "state"}};
};

} // namespace ops
} // namespace caffe2

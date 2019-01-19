#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"
#include <ATen/core/blob.h>
#include <ATen/core/dispatch/OpSchema.h>

namespace caffe2 {
namespace ops {

struct BatchMatmul final {
  static constexpr const char* name = "batch_matmul";

  using Signature = void(
      const at::Tensor& A,
      const at::Tensor& B,
      const at::Tensor& output,
      int trans_a,
      int trans_b,
      int broadcast);

  static constexpr size_t num_dispatch_args() {return 2;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 6> parameter_names = {
      {"A",
       "B",
       "output",
       "trans_a",
       "trans_b",
       "broadcast"}};
};

} // namespace ops
} // namespace caffe2

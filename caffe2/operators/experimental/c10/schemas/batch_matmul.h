#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include "caffe2/core/context_base.h"
#include <ATen/core/blob.h>

namespace caffe2 {
namespace ops {

struct BatchMatmul final {
  struct State final {
    std::shared_ptr<at::Tensor> scratch;
  };

  static constexpr const char* name = "batch_matmul";

  using Signature = void(
      const at::Tensor& A,
      const at::Tensor& B,
      const at::Tensor& output,
      int trans_a,
      int trans_b,
      int broadcast,
      intrusive_ptr<Blob> state);

  static constexpr size_t num_dispatch_args() {return 2;}

  static constexpr size_t num_outputs() {return 1;}

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"A",
       "B",
       "output",
       "trans_a",
       "trans_b",
       "broadcast",
       "state"}};
};

} // namespace ops
} // namespace caffe2

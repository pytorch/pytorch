#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct BatchMatmul final {
  struct State final {
    std::shared_ptr<Tensor> scratch;
  };

  static constexpr const char* name = "batch_matmul";

  using Signature = void(
      const Tensor& A,
      const Tensor& B,
      Tensor* output,
      int trans_a,
      int trans_b,
      int broadcast,
      State* state,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 8> parameter_names = {
      {"A",
       "B",
       "output",
       "trans_a",
       "trans_b",
       "broadcast",
       "state",
       "context"}};
};

} // namespace ops
} // namespace caffe2

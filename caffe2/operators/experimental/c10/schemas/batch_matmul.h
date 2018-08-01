#pragma once

#include "caffe2/core/tensor.h"
#include "caffe2/utils/Array.h"

namespace caffe2 {
namespace ops {

struct BatchMatmul final {
  struct State final {
    std::shared_ptr<Tensor<CPUContext>> scratch;
  };

  static constexpr const char* name = "batch_matmul";

  using Signature = void(
      const Tensor<CPUContext>& A,
      const Tensor<CPUContext>& B,
      Tensor<CPUContext>* output,
      int trans_a,
      int trans_b,
      int broadcast,
      int use_scratch,
      State* state,
      CPUContext* context);

  static constexpr c10::guts::array<const char*, 9> parameter_names = {
      {"A",
       "B",
       "output",
       "trans_a",
       "trans_b",
       "broadcast",
       "use_scratch",
       "state",
       "context"}};
};

} // namespace ops
} // namespace caffe2

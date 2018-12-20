#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct ExpandDims final {
  struct State {
    std::vector<int> dims;
    bool initialized = false;
  };

  static constexpr const char* name = "expand_dims";

  using Signature = void(
      const Tensor& input,
      Tensor* output,
      const std::vector<int>& dims,
      State* state,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 5> parameter_names = {
      {"input", "output", "dims", "state", "context"}};
};

} // namespace ops
} // namespace caffe2

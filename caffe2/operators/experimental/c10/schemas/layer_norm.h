#pragma once

#include "caffe2/core/tensor.h"
#include <c10/util/Array.h>

namespace caffe2 {
namespace ops {

struct LayerNorm final {
  static constexpr const char* name = "LayerNorm";

  struct Cache final {
    Tensor scale = empty({}, CPU);
    Tensor bias = empty({}, CPU);
  };

  using Signature = void(
      const Tensor& input,
      Tensor* output,
      Tensor* output_mean,
      Tensor* output_stddev,
      int axis,
      float epsilon,
      Cache* cache,
      BaseContext* context);

  static constexpr c10::guts::array<const char*, 8> parameter_names = {
      {"input", "output", "output_mean", "output_stddev", "axis", "epsilon", "cache", "context"}};
};

} // namespace ops
} // namespace caffe2

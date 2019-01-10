#pragma once

#include <c10/core/Tensor.h>
#include <c10/util/Array.h>

namespace at {
class BaseContext;
}

namespace c10 {
namespace core {
namespace opschema {

// TODO This op schema should probably not live in c10 since it's not a method
// on Tensor. It's only here as a proof-of-concept op and for LATTE team
// to be able to call caffe2 layer norm from PyTorch.
struct LayerNorm final {
  static constexpr const char* name = "LayerNorm";

  struct Cache final {
      at::optional<C10Tensor> scale = at::nullopt;
      at::optional<C10Tensor> bias = at::nullopt;
  };

  using Signature = void(
      const C10Tensor& input,
      const C10Tensor& output,
      const C10Tensor& output_mean,
      const C10Tensor& output_stddev,
      int axis,
      float epsilon,
      Cache* cache,
      at::BaseContext* context);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 3;}

  static constexpr c10::guts::array<const char*, 8> parameter_names = {
      {"input", "output", "output_mean", "output_stddev", "axis", "epsilon", "cache", "context"}};
};

} // namespace opschema
} // namespace core
} // namespace c10

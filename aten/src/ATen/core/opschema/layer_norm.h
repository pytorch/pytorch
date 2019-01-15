#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Array.h>
#include <ATen/core/blob.h>

namespace c10 {
namespace core {
namespace opschema {

// TODO This op schema should probably not live in c10 since it's not a method
// on Tensor. It's only here as a proof-of-concept op and for LATTE team
// to be able to call caffe2 layer norm from PyTorch.
struct LayerNorm final {
  static constexpr const char* name = "LayerNorm";

  struct Cache final {
      at::optional<at::Tensor> scale = at::nullopt;
      at::optional<at::Tensor> bias = at::nullopt;
  };

  using Signature = void(
      const at::Tensor& input,
      const at::Tensor& output,
      const at::Tensor& output_mean,
      const at::Tensor& output_stddev,
      int axis,
      float epsilon,
      intrusive_ptr<caffe2::Blob> cache);

  static constexpr size_t num_dispatch_args() {return 1;}

  static constexpr size_t num_outputs() {return 3;}

  static constexpr c10::guts::array<const char*, 7> parameter_names = {
      {"input", "output", "output_mean", "output_stddev", "axis", "epsilon", "cache"}};
};

} // namespace opschema
} // namespace core
} // namespace c10

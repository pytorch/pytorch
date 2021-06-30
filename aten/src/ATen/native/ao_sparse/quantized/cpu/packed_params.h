#pragma once

#include <cstdint>

#include <ATen/core/ivalue.h>

namespace ao {
namespace sparse {

// <Weight, bias, out_features_block_size, in_features_block_size>
using LinearPackedSerializationType =
    std::tuple<at::Tensor, c10::optional<at::Tensor>, std::vector<int64_t>>;

struct LinearPackedParamsBase : public torch::jit::CustomClassHolder {
 public:
  LinearPackedParamsBase(
      const int64_t out_features_block_size,
      const int64_t in_features_block_size)
      : out_features_block_size_(out_features_block_size),
        in_features_block_size_(in_features_block_size) {}

  virtual at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;
  virtual at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) = 0;

  virtual at::Tensor apply_dynamic(const at::Tensor& input) = 0;
  virtual at::Tensor apply_dynamic_relu(const at::Tensor& input) = 0;

  virtual LinearPackedSerializationType unpack() = 0;

  virtual c10::optional<at::Tensor> bias() = 0;

  virtual void set_bias(const c10::optional<at::Tensor>& bias) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }

 protected:
  const int64_t out_features_block_size_, in_features_block_size_;
};

}}  // namespace ao::sparse

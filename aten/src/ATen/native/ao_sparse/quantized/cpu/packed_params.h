#pragma once

#include <cstdint>

#include <ATen/core/ivalue.h>

namespace ao::sparse {

// <Weight, bias, out_features_block_size, in_features_block_size>
using LinearPackedSerializationType =
    std::tuple<at::Tensor, std::optional<at::Tensor>, std::vector<int64_t>>;

#define SPARSE_LINEAR_PACKED_PARAM_SERIALIZATION_VERSION 2

using BCSRSerializationType =
    std::tuple<
        int64_t,                    // Serialization Version
        std::optional<at::Tensor>,  // Bias
        int64_t,                    // Out Features (Row) Block Size
        int64_t,                    // In Features (Column) Block Size
        at::Tensor,                 // Weight Scales (single element vector if per-tensor) (float)
        at::Tensor,                 // Wrapper for Weight Zero Points (single element vector if per-tensor) (int8_t)
        bool,                       // Quantization Scheme (true: per tensor, false: per channel)
        at::Tensor,                 // Wrapper for Row Block Indices (int8_t, int16_t, or int32_t)
        at::Tensor,                 // Wrapper for Column Block Indices (int8_t, int16_t, or int32_t)
        at::Tensor,                 // Wrapper for Non-Zero Weight Values, each +128 (uint8_t)
        int64_t,                    // Number of Output Channels
        int64_t                     // Number of Input Channels
    >;

using BCSR =
    std::tuple<
        std::vector<int8_t>,    // Non-Zero Weight Values
        std::vector<int32_t>,   // Compressed Row Block Indices
        std::vector<int32_t>    // Column Block Indices
    >;

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

  virtual BCSRSerializationType serialize() = 0;

  virtual std::optional<at::Tensor> bias() = 0;

  virtual void set_bias(const std::optional<at::Tensor>& bias) {
    throw std::runtime_error(
        "set_bias is not implemented for this packed "
        "parameter type");
  }

 protected:
  const int64_t out_features_block_size_, in_features_block_size_;
};

}  // namespace ao::sparse

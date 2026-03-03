#pragma once

#include <ATen/BlasBackend.h>
#include <ATen/core/Tensor.h>

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace at::native::scaled {

/**
 * Track concrete implementations available
 */
enum class ScaledGemmImplementation {
  NONE = 0,
  TENSORWISE_TENSORWISE = 1,
  ROWWISE_ROWWISE = 2,
  BLOCK_128x128_1x128 = 3,
  BLOCK_1x128_128x128 = 4,
  BLOCK_1x128_1x128 = 5,
  MXFP8_MXFP8 = 6,
  NVFP4_NVFP4 = 7,
  NVFP4_NVFP4_SINGLE_SCALE = 8,
  MXFP4_MXFP4 = 9,
};

/**
 * Convert passed int (enum) from python back into a
 * strictly-typed enum
 */
template <class EnumType, class ArrayType>
std::vector<EnumType> convert_int_to_enum(ArrayType& v) {
  std::vector<EnumType> converted;
  converted.reserve(v.size());

  for (auto vi : v) {
    converted.push_back(static_cast<EnumType>(vi));
  }
  return converted;
}

/**
 * Both inputs must be fp8,
 * Each needs a single scale, {Tensorwise (float)}
 */
TORCH_API
bool check_tensorwise_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b);

/**
 * Both inputs must be fp8,
 * Each needs scales, {Rowwise (float)}
 */
TORCH_API
bool check_rowwise_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b);

/**
 * Two-level scaling, canonical NVFP4
 * Both inputs must be fp4
 * A, B need 2 scales, {Blockwise_1x16 (e4m3), Tensorwise (fp32)}
 */
TORCH_API
bool check_nvfp4_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b);

/**
 * Single-level scaling, what PyT currently understands
 * Both inputs must be fp4
 * A, B need 1 scale, {Blockwise_1x16 (e4m3)}
 */
TORCH_API
bool check_nvfp4_recipe_single_scale(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b);

/**
 * Both inputs must be fp8
 * A, B must only have 1 scale each, A: {Blockwise_1x128 (float), B:
 * {Blockwise_128x128 (float)
 */
TORCH_API
bool check_deepseek_recipe(
    ScalingType expected_recipe_a,
    ScalingType expected_recipe_b,
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b);

/**
 * Both inputs must be fp8
 * A, B must have 1 scale each, {Blockwise_1x32, e8m0}
 */
TORCH_API
bool check_mxfp8_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b);

/**
 * Both inputs must be fp4
 * A, B must have 1 scale each, {Blockwise_1x32, e8m0}
 */
TORCH_API
bool check_mxfp4_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b);

} // namespace at::scaled

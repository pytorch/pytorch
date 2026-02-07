#pragma once

#include <ATen/BlasBackend.h>
#include <ATen/core/Tensor.h>

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace at::scaled {

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
inline bool check_tensorwise_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (!isFloat8Type(type_a) || !isFloat8Type(type_b)) {
    return false;
  }

  // 1 scale each, {Tensorwise, float}
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }
  // Need {Blockwise_1x32, e8m0} for A & B
  if (recipe_a[0] != ScalingType::TensorWise) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != ScalingType::TensorWise) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float) return false;

  return true;
}

/**
 * Both inputs must be fp8,
 * Each needs scales, {Rowwise (float)}
 */
inline bool check_rowwise_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (!isFloat8Type(type_a) || !isFloat8Type(type_b)) {
    return false;
  }

  // 1 scale each, {Tensorwise, float}
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {RowWise, dp32} for A & B
  if (recipe_a[0] != ScalingType::RowWise) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != ScalingType::RowWise) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float) return false;

  return true;
}

/**
 * Two-level scaling, canonical NVFP4
 * Both inputs must be fp4
 * A, B need 2 scales, {Blockwise_1x16 (e4m3), Tensorwise (fp32)}
 */
inline bool check_nvfp4_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // both types must be fp4
  if (type_a != ScalarType::Float4_e2m1fn_x2 || type_b != ScalarType::Float4_e2m1fn_x2) {
    return false;
  }

  // 2 scales, 2 recipes for each input
  if (scales_a.size() != 2 || recipe_a.size() != 2 || scales_b.size() != 2 || recipe_b.size() != 2) {
    return false;
  }

  // Need {Blockwise_1x16, e4m3 for scale[0], Tensorwise, fp32 for scale[1]}
  if (recipe_a[0] != ScalingType::BlockWise1x16 || recipe_a[1] != ScalingType::TensorWise) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e4m3fn || scales_a[1].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x16 || recipe_b[1] != ScalingType::TensorWise) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e4m3fn || scales_b[1].scalar_type() != ScalarType::Float) return false;

  return true;
}

/**
 * Single-level scaling, what PyT currently understands
 * Both inputs must be fp4
 * A, B need 1 scale, {Blockwise_1x16 (e4m3)}
 */
inline bool check_nvfp4_recipe_single_scale(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // both types must be fp4
  if (type_a != ScalarType::Float4_e2m1fn_x2 || type_b != ScalarType::Float4_e2m1fn_x2) {
    return false;
  }

  // 2 scales, 2 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x16, e4m3 for scale[0], Tensorwise, fp32 for scale[1]}
  if (recipe_a[0] != ScalingType::BlockWise1x16) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e4m3fn) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x16) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e4m3fn) return false;

  return true;
}

/**
 * Both inputs must be fp8
 * A, B must only have 1 scale each, A: {Blockwise_1x128 (float), B:
 * {Blockwise_128x128 (float)
 */
inline bool check_deepseek_recipe(
    ScalingType expected_recipe_a,
    ScalingType expected_recipe_b,
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (type_a != ScalarType::Float8_e4m3fn || type_b != ScalarType::Float8_e4m3fn) {
    return false;
  }

  // 1 scales, 1 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x128, float} for A, {Blockwise_128x128, float} for B
  if (recipe_a[0] != expected_recipe_a) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float) return false;
  if (recipe_b[0] != expected_recipe_b) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float) return false;

  return true;
}

/**
 * Both inputs must be fp8
 * A, B must have 1 scale each, {Blockwise_1x32, e8m0}
 */
inline bool check_mxfp8_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // both types must be fp8
  if (type_a != ScalarType::Float8_e4m3fn || type_b != ScalarType::Float8_e4m3fn) {
    return false;
  }

  // 1 scales, 1 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x32, e8m0} for A & B
  if (recipe_a[0] != ScalingType::BlockWise1x32) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x32) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;

  return true;
}

/**
 * Both inputs must be fp4
 * A, B must have 1 scale each, {Blockwise_1x32, e8m0}
 */
inline bool check_mxfp4_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // both types must be fp4
  if (type_a != ScalarType::Float4_e2m1fn_x2 || type_b != ScalarType::Float4_e2m1fn_x2) {
    return false;
  }

  // 1 scales, 1 recipes for each input
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // Need {Blockwise_1x32, e8m0} for A & B
  if (recipe_a[0] != ScalingType::BlockWise1x32) return false;
  if (scales_a[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;
  if (recipe_b[0] != ScalingType::BlockWise1x32) return false;
  if (scales_b[0].scalar_type() != ScalarType::Float8_e8m0fnu) return false;

  return true;
}

} // namespace at::scaled

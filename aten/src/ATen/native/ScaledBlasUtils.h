#pragma once

#include <ATen/BlasBackend.h>
#include <ATen/core/Tensor.h>

#include <array>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

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
 * Signature of the per-recipe acceptance functions in this header
 * (check_tensorwise_recipe, check_rowwise_recipe, ...). Each backend
 * builds an array of (name, acceptance_fn, ScaledGemmImplementation)
 * to enumerate the scaling configurations it supports.
 */
using acceptance_fn = std::function<bool(
    c10::ScalarType,
    std::vector<ScalingType>&,
    c10::ArrayRef<Tensor>&,
    c10::ScalarType,
    std::vector<ScalingType>&,
    c10::ArrayRef<Tensor>&)>;

using ScaleKernelDispatchEntry =
    std::tuple<std::string, acceptance_fn, ScaledGemmImplementation>;

/**
 * Walk a backend's dispatch table and return the first
 * ScaledGemmImplementation whose acceptance function matches the inputs.
 * Returns ScaledGemmImplementation::NONE if no entry matches; the caller
 * is responsible for raising a backend-appropriate error in that case.
 */
template <size_t N>
ScaledGemmImplementation find_scaled_gemm_impl(
    const std::array<ScaleKernelDispatchEntry, N>& dispatch,
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    c10::ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    c10::ArrayRef<Tensor>& scales_b) {
  for (const auto& fn_entry : dispatch) {
    const auto& accept_fn = std::get<1>(fn_entry);
    if (accept_fn(type_a, recipe_a, scales_a, type_b, recipe_b, scales_b)) {
      return std::get<2>(fn_entry);
    }
  }
  return ScaledGemmImplementation::NONE;
}

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

/**
 * Validate v2 _scaled_mm inputs and per-recipe scale shapes/dtypes.
 * Centralized here so it can be called from both TORCH_META_FUNC (so
 * `torch.compile` tracing fails fast with a precise message) and the
 * eager IMPL_FUNCs.
 */
TORCH_API
void validate_scaled_mm_v2_inputs(
    const Tensor& mat_a,
    const Tensor& mat_b,
    ArrayRef<Tensor> scale_a,
    ArrayRef<ScalingType> recipe_a,
    ArrayRef<SwizzleType> swizzle_a,
    ArrayRef<Tensor> scale_b,
    ArrayRef<ScalingType> recipe_b,
    ArrayRef<SwizzleType> swizzle_b);

} // namespace at::scaled

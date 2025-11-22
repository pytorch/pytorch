#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/util/typeid.h>
#include <cstdint>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/BlasBackend.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>

#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUScaledBlas.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

using at::blas::ScalingType;

namespace at::native::onednn::scaled {

/**
 * Both inputs must be fp8,
 * Each needs a single scale, {Tensorwise (float)}
 */
bool check_tensorwise_recipe(
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
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 ||
      recipe_b.size() != 1) {
    return false;
  }
  // Need {Blockwise_1x32, e8m0} for A & B
  if (recipe_a[0] != ScalingType::TensorWise)
    return false;
  if (scales_a[0].scalar_type() != ScalarType::Float)
    return false;
  if (recipe_b[0] != ScalingType::TensorWise)
    return false;
  if (scales_b[0].scalar_type() != ScalarType::Float)
    return false;

  return true;
}

/**
 * Both inputs must be fp8,
 * Each needs scales, {Rowwise (float)}
 */
bool check_rowwise_recipe(
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
  if (scales_a.size() != 1 || recipe_a.size() != 1 || scales_b.size() != 1 ||
      recipe_b.size() != 1) {
    return false;
  }

  // Need {RowWise, dp32} for A & B
  if (recipe_a[0] != ScalingType::RowWise)
    return false;
  if (scales_a[0].scalar_type() != ScalarType::Float)
    return false;
  if (recipe_b[0] != ScalingType::RowWise)
    return false;
  if (scales_b[0].scalar_type() != ScalarType::Float)
    return false;

  return true;
}

/**
 * A must be fp16/bf16, B must be fp8,
 * A needs no scale (implicitly 1.0),
 * B needs a single scale {Tensorwise (float)}
 */
bool check_a16w8_tensorwise_recipe(
    c10::ScalarType type_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // A must be fp16 or bf16, B must be fp8
  if ((type_a != ScalarType::Half && type_a != ScalarType::BFloat16) ||
      !isFloat8Type(type_b)) {
    return false;
  }

  // B should have 1 scale {Tensorwise, float}
  if (scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // B uses Tensorwise with float scale
  if (recipe_b[0] != ScalingType::TensorWise)
    return false;
  if (scales_b[0].scalar_type() != ScalarType::Float)
    return false;

  return true;
}

/**
 * A must be fp16/bf16, B must be fp8,
 * A needs no scale (implicitly 1.0), B needs a rowwise scale {RowWise (float)}
 */
bool check_a16w8_rowwise_recipe(
    c10::ScalarType type_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // A must be fp16 or bf16, B must be fp8
  if ((type_a != ScalarType::Half && type_a != ScalarType::BFloat16) ||
      !isFloat8Type(type_b)) {
    return false;
  }

  // B should have 1 scale {Rowwise, float}
  if (scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // B uses Rowwise with float scale
  if (recipe_b[0] != ScalingType::RowWise)
    return false;
  if (scales_b[0].scalar_type() != ScalarType::Float)
    return false;

  return true;
}

/**
 * A must be fp16/bf16, B must be fp8,
 * A needs no scale (implicitly 1.0),
 * B needs a channelwise scale {ChannelWise (float)}
 */
bool check_a16w8_channelwise_recipe(
    c10::ScalarType type_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    ArrayRef<Tensor>& scales_b) {
  // A must be fp16 or bf16, B must be fp8
  if ((type_a != ScalarType::Half && type_a != ScalarType::BFloat16) ||
      !isFloat8Type(type_b)) {
    return false;
  }

  // B should have 1 scale {Channelwise, float}
  if (scales_b.size() != 1 || recipe_b.size() != 1) {
    return false;
  }

  // B uses Channelwise with float scale
  if (recipe_b[0] != ScalingType::ChannelWise)
    return false;
  if (scales_b[0].scalar_type() != ScalarType::Float)
    return false;

  return true;
}

} // namespace at::native::onednn::scaled

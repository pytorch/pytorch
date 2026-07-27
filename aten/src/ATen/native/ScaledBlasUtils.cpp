#include <cstdint>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/BlasBackend.h>
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/ScaledBlasUtils.h>
#include <ATen/ceil_div.h>

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
#include <ATen/ops/relu.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#endif

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace at::native::scaled {

/**
 * Both inputs must be fp8,
 * Each needs a single scale, {Tensorwise (float)}
 */
bool check_tensorwise_recipe(
    c10::ScalarType type_a,
    std::vector<ScalingType>& recipe_a,
    c10::ArrayRef<Tensor>& scales_a,
    c10::ScalarType type_b,
    std::vector<ScalingType>& recipe_b,
    c10::ArrayRef<Tensor>& scales_b) {
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
bool check_nvfp4_recipe(
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
bool check_nvfp4_recipe_single_scale(
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
bool check_deepseek_recipe(
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
bool check_mxfp8_recipe(
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
bool check_mxfp4_recipe(
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

namespace {

bool is_fp8_or_fp4_type(ScalarType dtype) {
  return isFloat8Type(dtype) || dtype == ScalarType::Float4_e2m1fn_x2;
}

bool is_fp4_type(ScalarType dtype) {
  return dtype == ScalarType::Float4_e2m1fn_x2;
}

bool is_single_recipe(
    ArrayRef<ScalingType> recipe_a,
    ArrayRef<ScalingType> recipe_b,
    ScalingType expected_a,
    ScalingType expected_b) {
  return recipe_a.size() == 1 && recipe_b.size() == 1 &&
      recipe_a[0] == expected_a && recipe_b[0] == expected_b;
}

bool is_two_level_nvfp4(
    ArrayRef<ScalingType> recipe_a,
    ArrayRef<ScalingType> recipe_b) {
  return recipe_a.size() == 2 && recipe_b.size() == 2 &&
      recipe_a[0] == ScalingType::BlockWise1x16 &&
      recipe_a[1] == ScalingType::TensorWise &&
      recipe_b[0] == ScalingType::BlockWise1x16 &&
      recipe_b[1] == ScalingType::TensorWise;
}

} // namespace

// Eager-and-meta validation for `_scaled_mm_v2`. Mirrors a subset of the
// kernel-side checks so torch.compile tracing fails fast with the same
// messages users get in eager. We deliberately *don't* duplicate checks
// that depend on device capability (e.g. layout permutations and the
// DeepSeek SM90 gate) -- those stay in the kernel so eager behavior
// matches the supported architectures.
//
// Uses sym_numel()/sym_size() so dynamic-shape tracing through Dynamo
// (which materializes scale tensors as fakes with symbolic dims) still
// goes through this path -- raw numel()/size() throw on symbolic shapes.
void validate_scaled_mm_v2_inputs(
    const Tensor& mat_a,
    const Tensor& mat_b,
    ArrayRef<Tensor> scale_a,
    ArrayRef<ScalingType> recipe_a,
    ArrayRef<SwizzleType> swizzle_a,
    ArrayRef<Tensor> scale_b,
    ArrayRef<ScalingType> recipe_b,
    ArrayRef<SwizzleType> swizzle_b) {
  TORCH_CHECK_VALUE(
      is_fp8_or_fp4_type(mat_a.scalar_type()) && is_fp8_or_fp4_type(mat_b.scalar_type()),
      "Expected both inputs to be fp8 or fp4 types but got mat_a.dtype=",
      mat_a.scalar_type(), " and mat_b.dtype=", mat_b.scalar_type());

  // mat_a: [M, K], mat_b: [K, N]. fp4 inputs are packed 2x along K, so the
  // unpacked K used by the blockwise scale-shape formulas is `2 * mat_a.size(1)`.
  const auto M = mat_a.sym_size(0);
  const auto N = mat_b.sym_size(1);
  const bool both_fp4 = is_fp4_type(mat_a.scalar_type()) && is_fp4_type(mat_b.scalar_type());
  const auto K_unpacked = both_fp4 ? mat_a.sym_size(1) * 2 : mat_a.sym_size(1);
  // XPU (oneDNN) uses unswizzled, unpadded blockwise scale shapes, unlike the
  // L4-padded SWIZZLE_32_4_4 layout that CUDA expects.
  const bool is_xpu = mat_a.is_xpu();

  auto sym_ceil_div = [](const c10::SymInt& a, int64_t b) {
    return (a + b - 1) / b;
  };
  auto sym_round_up = [&](const c10::SymInt& a, int64_t b) {
    return sym_ceil_div(a, b) * b;
  };

  // XPU (oneDNN) also accepts a column-major (transpose-contiguous) layout
  // because it calls .contiguous() internally; CUDA/ROCm require a plain
  // contiguous scale.
  auto is_scale_contiguous = [is_xpu](const Tensor& s) {
    return is_xpu ? (s.is_contiguous() || s.t().is_contiguous())
                  : s.is_contiguous();
  };

  // NVFP4 (BlockWise1x16) blockwise scale element count for a given outer
  // dim, shared by the single- and two-level NVFP4 recipes. XPU uses the
  // unpadded shape; NVIDIA the L4-padded SWIZZLE_32_4_4 shape. ROCm doesn't
  // support NVFP4, so only these two cases arise.
  auto nvfp4_scale_elems = [&](const c10::SymInt& dim) {
    return is_xpu ? dim * sym_ceil_div(K_unpacked, 16)
                  : sym_round_up(dim, 128) *
            sym_round_up(sym_ceil_div(K_unpacked, 16), 4);
  };

  // Per-recipe scale checks. Shapes and dtypes mirror the per-recipe
  // acceptance functions in aten/src/ATen/native/cuda/ScaledBlas.cpp so a
  // configuration that survives the meta also survives the kernel
  // dispatch's `find_scaled_gemm_impl` table walk.
  const bool is_tw = is_single_recipe(
      recipe_a, recipe_b, ScalingType::TensorWise, ScalingType::TensorWise);
  const bool is_rw = is_single_recipe(
      recipe_a, recipe_b, ScalingType::RowWise, ScalingType::RowWise);
  const bool is_mx_1x32 = is_single_recipe(
      recipe_a, recipe_b, ScalingType::BlockWise1x32, ScalingType::BlockWise1x32);
  const bool is_nv_1x16 = is_single_recipe(
      recipe_a, recipe_b, ScalingType::BlockWise1x16, ScalingType::BlockWise1x16);
  const bool is_nv_2lvl = is_two_level_nvfp4(recipe_a, recipe_b);
  // BlockWise1x128/128x128 combinations (DeepSeek-style) are deliberately
  // not validated here: they require SM90 and fail with NotImplementedError
  // from the kernel on other archs, which tests rely on.
  const bool is_deepseek = is_single_recipe(
          recipe_a, recipe_b, ScalingType::BlockWise1x128, ScalingType::BlockWise1x128) ||
      is_single_recipe(
          recipe_a, recipe_b, ScalingType::BlockWise1x128, ScalingType::BlockWise128x128) ||
      is_single_recipe(
          recipe_a, recipe_b, ScalingType::BlockWise128x128, ScalingType::BlockWise1x128);

  if (is_tw) {
    TORCH_CHECK_VALUE(
        scale_a.size() == 1 && scale_a[0].sym_numel() == 1 &&
            scale_a[0].scalar_type() == ScalarType::Float,
        "scale_a must have 1 Float element");
    TORCH_CHECK_VALUE(
        scale_b.size() == 1 && scale_b[0].sym_numel() == 1 &&
            scale_b[0].scalar_type() == ScalarType::Float,
        "scale_b must have 1 Float element");
  } else if (is_rw) {
    // Match the kernel's per-tensor RowWise wording exactly. We don't check
    // dim()/contiguity here -- the kernel has specific, more actionable
    // diagnostics for those (e.g. "expected scale_b.stride(1) to be 1...").
    TORCH_CHECK_VALUE(
        scale_a.size() == 1 && scale_a[0].sym_numel() == M &&
            scale_a[0].scalar_type() == ScalarType::Float,
        "scale_a must have ", M, " Float elements, got ",
        scale_a.empty() ? c10::SymInt(0) : scale_a[0].sym_numel());
    TORCH_CHECK_VALUE(
        scale_b.size() == 1 && scale_b[0].sym_numel() == N &&
            scale_b[0].scalar_type() == ScalarType::Float,
        "scale_b must have ", N, " Float elements, got ",
        scale_b.empty() ? c10::SymInt(0) : scale_b[0].sym_numel());
  } else if (is_mx_1x32) {
    c10::SymInt expected_a_elems;
    c10::SymInt expected_b_elems;
    // ROCm and NVIDIA use different blockwise scale shapes; detect at runtime
    // to keep aten-cpu free of GPU-conditional compilation. Formulas mirror
    // _scaled_mxfp8_mxfp8 and _scaled_mxfp4_mxfp4 in cuda/ScaledBlas.cpp.
    if (is_xpu) {
      // XPU: unpadded [M, ceil_div(K, 32)] / [N, ceil_div(K, 32)], NO_SWIZZLE.
      expected_a_elems = M * sym_ceil_div(K_unpacked, 32);
      expected_b_elems = N * sym_ceil_div(K_unpacked, 32);
    } else if (at::globalContext().hasROCM()) {
      // ROCm: K_multiplier=2 doubles M and N (the non-contraction dims) for
      // packed fp4; K (= mat_a.sym_size(1)) is used raw on both sides.
      const auto K = mat_a.sym_size(1);
      const auto M_eff = both_fp4 ? M * 2 : M;
      const auto N_eff = both_fp4 ? N * 2 : N;
      expected_a_elems = sym_ceil_div(M_eff, 32) * K;
      expected_b_elems = sym_ceil_div(N_eff, 32) * K;
    } else {
      // NVIDIA: K_unpacked is doubled for packed fp4 (K_multiplier=2).
      expected_a_elems = sym_round_up(M, 128) *
          sym_round_up(sym_ceil_div(K_unpacked, 32), 4);
      expected_b_elems = sym_round_up(N, 128) *
          sym_round_up(sym_ceil_div(K_unpacked, 32), 4);
    }
    TORCH_CHECK_VALUE(
        scale_a.size() == 1 && scale_a[0].sym_numel() == expected_a_elems &&
            scale_a[0].scalar_type() == ScalarType::Float8_e8m0fnu,
        "For Blockwise scaling scale_a should have ", expected_a_elems,
        " elements, got: ", scale_a.empty() ? c10::SymInt(0) : scale_a[0].sym_numel());
    TORCH_CHECK_VALUE(
        scale_b.size() == 1 && scale_b[0].sym_numel() == expected_b_elems &&
            scale_b[0].scalar_type() == ScalarType::Float8_e8m0fnu,
        "For Blockwise scaling scale_b should have ", expected_b_elems,
        " elements, got: ", scale_b.empty() ? c10::SymInt(0) : scale_b[0].sym_numel());
  } else if (is_nv_1x16) {
    const auto expected_a_elems = nvfp4_scale_elems(M);
    const auto expected_b_elems = nvfp4_scale_elems(N);
    TORCH_CHECK_VALUE(
        scale_a.size() == 1 && scale_a[0].sym_numel() == expected_a_elems &&
            scale_a[0].scalar_type() == ScalarType::Float8_e4m3fn,
        "For Blockwise scaling scale_a should have ", expected_a_elems,
        " elements, got: ", scale_a.empty() ? c10::SymInt(0) : scale_a[0].sym_numel());
    TORCH_CHECK_VALUE(
        scale_b.size() == 1 && scale_b[0].sym_numel() == expected_b_elems &&
            scale_b[0].scalar_type() == ScalarType::Float8_e4m3fn,
        "For Blockwise scaling scale_b should have ", expected_b_elems,
        " elements, got: ", scale_b.empty() ? c10::SymInt(0) : scale_b[0].sym_numel());
  } else if (is_nv_2lvl) {
    const auto expected_a_elems = nvfp4_scale_elems(M);
    const auto expected_b_elems = nvfp4_scale_elems(N);
    // Split the count check from the per-element check so the error
    // message points at the actual mismatch.
    TORCH_CHECK_VALUE(
        scale_a.size() == 2 && scale_b.size() == 2,
        "Two-level NVFP4 scaling requires 2 scale tensors per side, got ",
        scale_a.size(), " and ", scale_b.size());
    TORCH_CHECK_VALUE(
        scale_a[0].sym_numel() == expected_a_elems &&
            scale_a[0].scalar_type() == ScalarType::Float8_e4m3fn &&
            scale_a[1].sym_numel() == 1 &&
            scale_a[1].scalar_type() == ScalarType::Float,
        "For Blockwise scaling scale_a should have ", expected_a_elems,
        " elements, got: ", scale_a[0].sym_numel());
    TORCH_CHECK_VALUE(
        scale_b[0].sym_numel() == expected_b_elems &&
            scale_b[0].scalar_type() == ScalarType::Float8_e4m3fn &&
            scale_b[1].sym_numel() == 1 &&
            scale_b[1].scalar_type() == ScalarType::Float,
        "For Blockwise scaling scale_b should have ", expected_b_elems,
        " elements, got: ", scale_b[0].sym_numel());
  } else if (!is_deepseek) {
    // Match the kernel's `find_scaled_gemm_impl` fall-through so unrecognized
    // recipe combinations fail at trace time rather than at kernel dispatch.
    TORCH_CHECK_VALUE(
        false,
        "Invalid scaling configuration for _scaled_mm_v2: unsupported recipe(s). "
        "Got mat_a.dtype=", mat_a.scalar_type(),
        ", mat_b.dtype=", mat_b.scalar_type(),
        ", recipe_a.size()=", recipe_a.size(),
        ", recipe_b.size()=", recipe_b.size());
  }

  // Swizzle count + value check, matching the kernel-side checks
  // `check_swizzle_lengths` and the per-impl `swizzle_a == ...` asserts in
  // aten/src/ATen/native/cuda/ScaledBlas.cpp. Only MX/NVFP recipes consult
  // swizzle; tensorwise/rowwise/deepseek paths don't.
  const bool is_mx_or_nvfp = is_mx_1x32 || is_nv_1x16 || is_nv_2lvl;
  if (is_mx_or_nvfp) {
    // Blockwise scales must be contiguous (checked once for all MX/NVFP4
    // recipes). scale_a[0]/scale_b[0] are the blockwise scales and were
    // size-validated in the per-recipe branches above.
    TORCH_CHECK_VALUE(
        is_scale_contiguous(scale_a[0]) && is_scale_contiguous(scale_b[0]),
        is_xpu
            ? "For Blockwise scaling both scales should be contiguous (row-major or column-major)"
            : "For Blockwise scaling both scales should be contiguous");
    const auto num_args_a = recipe_a.size();
    const auto num_args_b = recipe_b.size();
    const bool is_rocm = at::globalContext().hasROCM();
    if (is_xpu) {
      // XPU consumes unswizzled scales for all MX/NVFP4 recipes.
      // swizzle_a/swizzle_b may be omitted (empty) by Python callers
      // (F.scaled_mm defaults swizzle to None); treat that as NO_SWIZZLE.
      // Otherwise they must match the recipe count and be all NO_SWIZZLE,
      // mirroring the XPU kernel.
      TORCH_CHECK_VALUE(
          (swizzle_a.empty() || swizzle_a.size() == num_args_a) &&
              (swizzle_b.empty() || swizzle_b.size() == num_args_b),
          "swizzle_a/swizzle_b must be empty or match the number of scale recipes, got ",
          swizzle_a.size(), " and ", swizzle_b.size());
      for (auto s : swizzle_a) {
        TORCH_CHECK_VALUE(
            s == SwizzleType::NO_SWIZZLE,
            "For XPU MX/NVFP4 gemm, scale_a swizzle entries must all be NO_SWIZZLE");
      }
      for (auto s : swizzle_b) {
        TORCH_CHECK_VALUE(
            s == SwizzleType::NO_SWIZZLE,
            "For XPU MX/NVFP4 gemm, scale_b swizzle entries must all be NO_SWIZZLE");
      }
    } else if (!is_rocm) {
      TORCH_CHECK_VALUE(
          swizzle_a.size() == num_args_a,
          "swizzle_a must have ", num_args_a, " value",
          num_args_a == 1 ? "" : "s",
          ", got ", swizzle_a.size());
      TORCH_CHECK_VALUE(
          swizzle_b.size() == num_args_b,
          "swizzle_b must have ", num_args_b, " value",
          num_args_b == 1 ? "" : "s",
          ", got ", swizzle_b.size());
      // The first swizzle slot is the blockwise scale's swizzle and must be
      // SWIZZLE_32_4_4 on NVIDIA. The second slot (NVFP4 two-level) is the
      // tensorwise scale and is unused, so don't check it.
      TORCH_CHECK_VALUE(
          swizzle_a[0] == SwizzleType::SWIZZLE_32_4_4,
          "scale_a must be swizzled to SWIZZLE_32_4_4 format");
      TORCH_CHECK_VALUE(
          swizzle_b[0] == SwizzleType::SWIZZLE_32_4_4,
          "scale_b must be swizzled to SWIZZLE_32_4_4 format");
    } else if (is_mx_1x32) {
      // ROCm only supports the MX path (no NVFP4) and requires NO_SWIZZLE.
      TORCH_CHECK_VALUE(
          swizzle_a.size() == 1 && swizzle_b.size() == 1,
          "For ROCM MX gemm, swizzle_a and swizzle_b must each have 1 value, got ",
          swizzle_a.size(), " and ", swizzle_b.size());
      TORCH_CHECK_VALUE(
          swizzle_a[0] == SwizzleType::NO_SWIZZLE &&
              swizzle_b[0] == SwizzleType::NO_SWIZZLE,
          "For ROCM MX gemm, swizzle_a and swizzle_b must both be NO_SWIZZLE");
    }
  }
}

}  // at::scaled

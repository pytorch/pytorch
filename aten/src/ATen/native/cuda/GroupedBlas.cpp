#include <cstdint>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAScaledBlas.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/GroupMM.h>
#ifdef USE_ROCM
#include <ATen/native/hip/ck_group_gemm.h>
#endif
#include <ATen/ceil_div.h>

#ifdef USE_FBGEMM_GENAI
#include <fbgemm_gpu/torch_ops.h>
#endif

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

namespace scaled_blas = at::cuda::scaled;
using scaled_blas::ScaledGemmImplementation;
using scaled_blas::convert_int_to_enum;
using scaled_blas::_scaled_mm_allowed_device;

namespace at::native {

namespace {

// 2d-2d and 2d-3d
// scaling=MXFP8
// CUDA-only
Tensor&
_mx8_mx8_bf16_grouped_mm_fbgemm(
        const Tensor& mat_a,
        const Tensor& mat_b,
        const Tensor& scale_a,
        const SwizzleType swizzle_a,
        const Tensor& scale_b,
        const SwizzleType swizzle_b,
        const std::optional<at::Tensor>& offs,
        Tensor& out) {
    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;
    bool b_is_3d = mat_b.dim() == 3;
    bool is_2d_2d = a_is_2d && b_is_2d;
    bool is_2d_3d = a_is_2d && b_is_3d;
    TORCH_CHECK_VALUE(is_2d_2d || is_2d_3d, "MXFP8 grouped GEMM currently only supports 2d-2d and 2d-3d cases");
    TORCH_CHECK_VALUE(offs.has_value(), "MXFP8 2d-2d and 2d-3d grouped GEMMs requires offsets");
    TORCH_CHECK_VALUE(out.scalar_type() == at::kBFloat16, "Only bf16 out_dtype is supported for MXFP8 grouped gemm");
    // MXFP8 expects float8_e8m0fnu scales.
    TORCH_CHECK_VALUE(scale_a.scalar_type() == at::kFloat8_e8m0fnu && scale_b.scalar_type() == at::kFloat8_e8m0fnu,
        "For MXFP8 grouped gemm, both scales must be float8_e8m0fnu tensors.");
#ifdef USE_ROCM
    TORCH_CHECK_VALUE(swizzle_a == SwizzleType::NO_SWIZZLE && swizzle_b == SwizzleType::NO_SWIZZLE,
        "For ROCM MXFP8 grouped gemm, both scale swizzle types must be SWIZZLE_NONE");
#else
    TORCH_CHECK_VALUE(swizzle_a == SwizzleType::SWIZZLE_32_4_4 && swizzle_b == SwizzleType::SWIZZLE_32_4_4,
        "For CUDA MXFP8 grouped gemm, both scale swizzle types must be SWIZZLE_32_4_4");
#endif

#if defined(USE_FBGEMM_GENAI) and !defined(USE_ROCM)
    fbgemm_gpu::mx8mx8bf16_grouped_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs.value(),
        out);
#else
    TORCH_CHECK_NOT_IMPLEMENTED(false, "mxfp8_mxfp8 grouped gemm requires compile with USE_FBGEMM_GENAI");
#endif
    return out;
}

// 2d-2d and 2d-3d cases
// scaling=rowwise
// CUDA-only
Tensor&
_f8_f8_bf16_rowwise_grouped_mm_cuda(
          const Tensor& mat_a,
          const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<Tensor>& offs,
          const std::optional<Tensor>& bias,
          const bool use_fast_accum,
          Tensor& out) {
  TORCH_CHECK_VALUE(mat_a.dtype() == at::kFloat8_e4m3fn, "Expected mat_a to be Float8_e4m3 matrix got ", mat_a.scalar_type());
  TORCH_CHECK_VALUE(mat_b.dtype() == at::kFloat8_e4m3fn, "Expected mat_a to be Float8_e4m3 matrix got ", mat_b.scalar_type());

  at::cuda::detail::f8f8bf16_grouped_mm(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      offs,
      bias,
      use_fast_accum,
      out);
    return out;
}

// 2d-2d and 2d-3d cases
// scaling=rowwise
// only being called for rocm
Tensor&
_f8_f8_bf16_rowwise_grouped_mm_rocm(
      const Tensor& mat_a,
      const Tensor& mat_b,
      const Tensor& scale_a,
      const Tensor& scale_b,
      const std::optional<Tensor>& offs,
      Tensor& out) {
  TORCH_CHECK_VALUE(mat_a.dtype() == at::kFloat8_e4m3fnuz, "Expected mat_a to be Float8_e4m3fnuz matrix got ", mat_a.scalar_type());
  TORCH_CHECK_VALUE(mat_b.dtype() == at::kFloat8_e4m3fnuz, "Expected mat_a to be Float8_e4m3fnuz matrix got ", mat_b.scalar_type());

#if defined(USE_FBGEMM_GENAI) && defined(USE_ROCM)
  fbgemm_gpu::f8f8bf16_rowwise_grouped_mm(
      mat_a,
      // FBGEMM expects B matrix shape to be (.., N, K)
      mat_b.transpose(-2, -1),
      scale_a,
      scale_b,
      offs,
      out);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, "grouped gemm is not supported without USE_FBGEMM_GENAI on ROCM")
#endif
  return out;

}

// Dispatch f8 x f8 -> bf16 row-wise scaled to rocm/cuda
Tensor&
_f8_f8_bf16_rowwise_grouped_mm(
      const Tensor& mat_a,
      const Tensor& mat_b,
      const Tensor& scale_a,
      const Tensor& scale_b,
      const std::optional<Tensor>& offs,
      const std::optional<Tensor>& bias,
      bool use_fast_accum,
      Tensor& out) {
  // FP8 per-tensor and per-row scaling expect fp32 scales.
  TORCH_CHECK_VALUE(scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat,
      "For grouped FP8 rowwise, both scales must be float32 tensors");
#ifndef USE_ROCM
  return _f8_f8_bf16_rowwise_grouped_mm_cuda(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      offs,
      bias,
      use_fast_accum,
      out);
#else
  // NOTE: ignore use_fast_accum
  TORCH_CHECK_VALUE(!bias.has_value(), "ROCM grouped gemm does not support bias")
  return _f8_f8_bf16_rowwise_grouped_mm_rocm(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      offs,
      out);
#endif
}

Tensor&
_f4_f4_bf16_grouped_mm_fbgemm(
      const Tensor& mat_a,
      const Tensor& mat_b,
      const Tensor& scale_a,
      const std::optional<Tensor>& global_scale_a,
      const Tensor& scale_b,
      const std::optional<Tensor>& global_scale_b,
      const std::optional<Tensor>& offs,
      const std::optional<Tensor>& bias,
      Tensor& out) {
#if !defined(USE_ROCM) && defined(USE_FBGEMM_GENAI)
  // Typing checks
  TORCH_CHECK_VALUE(mat_a.scalar_type() == at::kFloat4_e2m1fn_x2,
      "mat_a must be Float4_e2n1fn_2, got: ", mat_a.scalar_type());
  TORCH_CHECK_VALUE(mat_b.scalar_type() == at::kFloat4_e2m1fn_x2,
      "mat_b must be Float4_e2n1fn_2, got: ", mat_b.scalar_type());

  std::optional<Tensor> combined_global_scale = std::nullopt;
  if (global_scale_a.has_value() || global_scale_b.has_value()) {
      // NVFP4
      TORCH_CHECK_VALUE(global_scale_a.has_value() && global_scale_b.has_value(),
          "For NVFP4 grouped gemm both of global_scale_{a,b} must have values")
      TORCH_CHECK_VALUE(scale_a.scalar_type() == at::kFloat8_e4m3fn,
          "scale_a must be Float8_e4m3fn, got: ", scale_a.scalar_type());
      TORCH_CHECK_VALUE(scale_b.scalar_type() == at::kFloat8_e4m3fn,
          "scale_b must be Float8_e4m3fn, got: ", scale_b.scalar_type());
      TORCH_CHECK_VALUE(global_scale_a.value().scalar_type() == at::kFloat,
          "global_scale_a must be Float, got: ", global_scale_a.value().scalar_type());
      TORCH_CHECK_VALUE(global_scale_b.value().scalar_type() == at::kFloat,
          "global_scale_b must be Float, got: ", global_scale_b.value().scalar_type());
      combined_global_scale = global_scale_a.value().mul(global_scale_b.value());
  } else {
      // MXFP4
      TORCH_CHECK_VALUE(scale_a.scalar_type() == at::kFloat8_e8m0fnu,
          "scale_a must be Float8_e8m0fnu, got: ", scale_a.scalar_type());
      TORCH_CHECK_VALUE(scale_b.scalar_type() == at::kFloat8_e8m0fnu,
          "scale_b must be Float8_e8m0fnu, got: ", scale_b.scalar_type());
  }

  auto o = fbgemm_gpu::f4f4bf16_grouped_mm(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      offs.value(),
      out,
      combined_global_scale
  );
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, "nvfp4 grouped gemm is not supported without USE_FBGEMM_GENAI, and only for CUDA")
#endif

  return out;
}

void _check_scales_fp8_rowwise(const Tensor& mat, const Tensor& scale, const int dim, const int arg_idx, const int scale_multiplier=1) {
  // Checks scales for 2d or 3d target tensors (`mat`).
  if (mat.dim() == 2) {
    TORCH_CHECK(
        scale.dim() == 1,
        "scale must be a 1D tensor, but got ",
        scale.dim(),
        "D, arg ",
        arg_idx);
    TORCH_CHECK(
        scale.is_contiguous(), "scale must be contiguous for arg ", arg_idx);
    TORCH_CHECK(
        scale.size(0) == mat.size(dim) * scale_multiplier,
        "scale must have the same length as mat for arg ",
        arg_idx);
  } else {
    TORCH_CHECK(
        scale.dim() == 2,
        "scale must be a 2D tensor, but got ",
        scale.dim(),
        "D for arg ",
        arg_idx);
    TORCH_CHECK(
        scale.stride(1) == 1,
        "scale must be contiguous in the last dimension for arg ",
        arg_idx);
    TORCH_CHECK(
        scale.size(0) == mat.size(0),
        "scale must have the same batch dimension as mat for arg ",
        arg_idx);
    TORCH_CHECK(
        scale.size(1) == mat.size(1 + dim),
        "scale must have the same first dimension as mat for arg ",
        arg_idx);
  }
}

void _check_scales_blocked(const Tensor& mat, const Tensor& scale, const int dim, const int arg_idx) {
  // if {mx,nv}fp4, will need to modify K later
  bool is_fp4 = (mat.scalar_type() == kFloat4_e2m1fn_x2);
  int blocksize = 32;
  // check for nvfp4 vs. mxfp4 to fix blocksize
  if (is_fp4 && scale.scalar_type() == kFloat8_e4m3fn) {
    blocksize = 16;
  }

  // Checks scales for 2d or 3d target tensors (`mat`).
  if (mat.dim() == 2) {
    // For MXFP8, 2d tensors have variable size groups represented as subtensors,
    // that are converted to blocked padded format individually,
    // so we can't check the scale sizes without doing a d2h sync to get the group sizes here.
    TORCH_CHECK(
      scale.dim() == mat.dim(),
      "for block-scaled, scale must have same number of dimensions as parent tensor, but got mat.dim() = ", mat.dim(),
      " and scale.dim() = ", scale.dim(), " for arg ", arg_idx
    );

    // LHS mat shape (M, total_K) -> scale shape (rounded_up(M, 128), rounded_up_per_group(K/blocksize, 4))
    // RHS mat shape (total_K, N) -> scale shape (rounded_up(N, 128), rounded_up_per_group(K/blocksize, 4))
    //   * weight is transposed prior to the call, scale stays non-transposed.
    bool LHS = arg_idx == 0;
    int scale_dim_to_check = 0;
    int mat_dim_to_check = LHS ? 0 : 1;
    TORCH_CHECK(
        scale.size(scale_dim_to_check) >= mat.size(mat_dim_to_check),
        "for block-scaled, arg ", arg_idx, " tensor shape (", mat.size(0), ", ", mat.size(1), ") ",
        "must have scale.shape[", scale_dim_to_check, "] >= ", mat.size(mat_dim_to_check), " but got scale.shape=(", scale.size(0), ", ", scale.size(1), ")");
  } else {
    // For MXFP8, 3d tensors have static group sizes (stack of 2d tensors),
    // so we can check the exact expected scale sizes here without a d2h sync.
    auto round_up = [](auto x, auto y) {
        return ((x + y - 1) / y) * y;
    };

    // TODO: this is for 3d tensor in 2d-3d case specifically.
    // We'll need to support 3d-3d and 3d-2d cases once mxfp8/nvfp4 grouped gemm supports them.
    int64_t G = mat.size(0);
    int64_t K = mat.size(1);
    if (is_fp4) {
      // FP4 packs 2 values into a single 8b word - the "real" K is 2x the
      // reported K. Reverse that adjustment.
      const int fp4_elems_per_byte = 2;
      K *= fp4_elems_per_byte;
    }
    int64_t N = mat.size(2);
    int64_t blocked_scale_K = round_up(K/blocksize, 4);
    int64_t blocked_scale_N = round_up(N, 128);

    // fbgemm expects stack of flattened blocked scales for 3d tensor, shape (G, blocked_scale_K * blocked_scale_N).
    TORCH_CHECK(
      scale.dim() == mat.dim() - 1,
      "for block-scaled 2d-3d grouped GEMM, the 3d tensor of shape (G,K,N) must have a 2d scale of shape (G, blocked_scale_K * blocked_scale_N),",
      "but scale is ", scale.dim(), "D for arg ", arg_idx
    );
    TORCH_CHECK(
      scale.size(0) == G && scale.size(1) == blocked_scale_K * blocked_scale_N,
      "for block-scaled grouped GEMM, the tensor shape (", G, ", ", K, ", ", N, ") must have scale shape (", G, ",", blocked_scale_K, ",", blocked_scale_N, ")",
      " for arg ", arg_idx, ", got: ", scale.size(0), ", ", scale.size(1)
    );
  }
}

void check_scale(const Tensor& mat, const Tensor& scale, const int dim, const int arg_idx, const int scale_multiplier=1) {
  bool using_fp8_rowwise = scale.scalar_type() == kFloat;
  bool using_mx = scale.scalar_type() == at::kFloat8_e8m0fnu;
  if (using_fp8_rowwise) {
    _check_scales_fp8_rowwise(mat, scale, dim, arg_idx, scale_multiplier);
  } else if (using_mx) {
    _check_scales_blocked(mat, scale, dim, arg_idx);
  } else {
    TORCH_CHECK(false, "scale must be float32 or float8_e8m0fnu, but got ", scale.dtype());
  }
}

} // namespace

Tensor
_scaled_grouped_mm_cuda(
        const Tensor& mat_a,
        const Tensor& mat_b,
        const Tensor& scale_a,
        const Tensor& scale_b,
        const std::optional<at::Tensor>& offs,
        const std::optional<at::Tensor>& bias,
        const std::optional<at::Tensor>& scale_result,
        std::optional<c10::ScalarType> out_dtype,
        bool use_fast_accum) {
  bool allowed_device = _scaled_mm_allowed_device(/*sm90_only*/true, /*sm100_only*/true);
  TORCH_CHECK_VALUE(allowed_device, "torch._scaled_grouped_mm is only supported on CUDA devices with compute capability = [9.0, 10.0], or ROCm MI300+");

  TORCH_CHECK_VALUE(!check_valid_strides_and_return_transposed(mat_a), "Expected mat1 to not be transposed");
  TORCH_CHECK_VALUE(check_valid_strides_and_return_transposed(mat_b), "Expected mat2 to be transposed");
  TORCH_CHECK_VALUE(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK_VALUE(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;

  // NOTE(slayton): For sub-1B formats want contraction_dim argument?
  if (!a_is_2d || !b_is_2d) {
    TORCH_CHECK_VALUE(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");
  }
  TORCH_CHECK_VALUE(
    mat_a.size(-1) % 16 == 0,
    "Expected trailing dimension of mat_a to be divisible by 16 ",
    "but got mat1 shape: (",
    mat_a.sizes(),
    ").");
  TORCH_CHECK_VALUE(mat_b.size(-2) % 16 == 0 && mat_b.size(-1) % 16 == 0,
    "Expected mat_b shape to be divisible by 16 ",
    "but got mat_b shape: (",
    mat_b.sizes(),
    ").");


  TORCH_CHECK_VALUE(!bias.has_value(), "Bias not supported yet");
  TORCH_CHECK_VALUE(!scale_result.has_value(), "Scale result not supported yet");
  TORCH_CHECK_VALUE(offs.has_value() ==  (a_is_2d || b_is_2d), "Have to provide offsets if there is a 2d matrix");

  // NOTE: mxfp8 x mxfp8 requires (and asserts later) that offsets is present.
  //       for rowwise, no offsets implies 3d-3d and is handled by lower-level
  //       routines
  if (offs.has_value()) {
    TORCH_CHECK_VALUE(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK_VALUE(offs->dtype() == at::kInt, "Offsets have to be int32");
  }
  // FP8 per-tensor and per-row scaling expect fp32 scales.
  // MXFP8 expects float8_e8m0fnu scales.
  TORCH_CHECK_VALUE(
      (scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat) ||
      (scale_a.scalar_type() == at::kFloat8_e8m0fnu && scale_b.scalar_type() == at::kFloat8_e8m0fnu),
      "For FP8 tensorwise and rowwise, both scales must both be float32 tensors. For MXFP8, scales must both be float8_e8m0fnu tensors.");

  const int scale_multiplier = (mat_a.dim() == 2 && mat_b.dim() == 2) ? offs->size(0) : 1;
  check_scale(mat_a, scale_a, 0 ,0, scale_multiplier);
  check_scale(mat_b, scale_b, 1, 1, scale_multiplier);

  const auto out_dtype_ = out_dtype.value_or(kBFloat16);
  TORCH_CHECK_VALUE(out_dtype_ == kBFloat16, "Only bf16 high precision output types are supported for grouped gemm");

  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

#if defined(USE_FBGEMM_GENAI) && defined(USE_CUDA) && !defined(USE_ROCM)
  // MXFP8 grouped GEMM dispatching
  bool is_mx8mx8bf16 = (
    mat_a.scalar_type() == at::kFloat8_e4m3fn && mat_b.scalar_type() == at::kFloat8_e4m3fn &&
    scale_a.scalar_type() == at::kFloat8_e8m0fnu && scale_b.scalar_type() == at::kFloat8_e8m0fnu
  );
#else
  bool is_mx8mx8bf16 = false;
#endif

  if (is_mx8mx8bf16) {
    // Note: Passing implied SwizzleType here, correctness of scale previously checked
    //       in `check_scale` call
    return _mx8_mx8_bf16_grouped_mm_fbgemm(
        mat_a,
        mat_b,
        scale_a,
        SwizzleType::SWIZZLE_32_4_4,
        scale_b,
        SwizzleType::SWIZZLE_32_4_4,
        offs.value(),
        out);
  }

  // If we're not MXFP8, then we're row-wise scaling.
  return _f8_f8_bf16_rowwise_grouped_mm(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      offs,
      bias,
      use_fast_accum,
      out);
}

namespace {

using acceptance_fn = std::function<bool(c10::ScalarType, std::vector<ScalingType>&, ArrayRef<Tensor>&, c10::ScalarType, std::vector<ScalingType>&, ArrayRef<Tensor>&)>;

std::array<std::tuple<std::string, acceptance_fn, ScaledGemmImplementation>, 4> scale_grouped_kernel_dispatch = {{
  { "rowwise_rowwise", scaled_blas::check_rowwise_recipe, ScaledGemmImplementation::ROWWISE_ROWWISE},
  { "mxfp8_mxfp8", scaled_blas::check_mxfp8_recipe, ScaledGemmImplementation::MXFP8_MXFP8},
  { "mxfp4_mxfp4", scaled_blas::check_mxfp4_recipe, ScaledGemmImplementation::MXFP4_MXFP4},
  { "nvfp4_nvfp4", scaled_blas::check_nvfp4_recipe, ScaledGemmImplementation::NVFP4_NVFP4}}};

} // anonymous namespace

Tensor
_scaled_grouped_mm_cuda_v2(
          const Tensor& mat_a, const Tensor& mat_b,
          ArrayRef<Tensor> scale_a,
          IntArrayRef scale_recipe_a,
          IntArrayRef swizzle_a,
          ArrayRef<Tensor> scale_b,
          IntArrayRef scale_recipe_b,
          IntArrayRef swizzle_b,
          const std::optional<Tensor>& offs,
          const std::optional<Tensor>& bias,
          const std::optional<c10::ScalarType> out_dtype,
          IntArrayRef contraction_dim,
          bool use_fast_accum) {
  bool allowed_device = _scaled_mm_allowed_device(/*sm90_only*/true, /*sm100_only*/true);
  TORCH_CHECK_VALUE(allowed_device, "torch._scaled_grouped_mm is only supported on CUDA devices with compute capability = [9.0, 10.0], or ROCm MI300+");

  TORCH_CHECK_VALUE(!check_valid_strides_and_return_transposed(mat_a), "Expected mat1 to not be transposed");
  TORCH_CHECK_VALUE(check_valid_strides_and_return_transposed(mat_b), "Expected mat2 to be transposed");
  TORCH_CHECK_VALUE(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK_VALUE(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;

  // NOTE(slayton): For sub-1B formats want contraction_dim argument?
  if (!a_is_2d || !b_is_2d) {
    if (contraction_dim.size() > 0) {
      const int dim_a = contraction_dim[0], dim_b = mat_b.size(contraction_dim[1]);
      TORCH_CHECK_VALUE(mat_a.size(dim_a) == mat_b.size(dim_b),
          "Contraction dimensions (", dim_a, ",", dim_b, ") of mat_a and mat_b must match, got: ", mat_a.size(dim_a), " and ",
          mat_b.size(dim_b));
      // Note: only (-1, -2) is currently supported
      TORCH_CHECK_VALUE(dim_a == -1 && dim_b == -2, "Curently contraction dims must be (-1, -2) only");
    } else {
      TORCH_CHECK_VALUE(mat_a.size(-1) == mat_b.size(-2), "contraction dimension of mat_a and mat_b must match");
    }
  }
  TORCH_CHECK_VALUE(
    mat_a.size(-1) % 16 == 0,
    "Expected trailing dimension of mat_a to be divisible by 16 ",
    "but got mat1 shape: (",
    mat_a.sizes(),
    ").");
  TORCH_CHECK_VALUE(mat_b.size(-2) % 16 == 0 && mat_b.size(-1) % 16 == 0,
    "Expected mat_b shape to be divisible by 16 ",
    "but got mat_b shape: (",
    mat_b.sizes(),
    ").");

  TORCH_CHECK_VALUE(!bias.has_value(), "Bias not supported yet");
  TORCH_CHECK_VALUE(offs.has_value() ==  (a_is_2d || b_is_2d), "Have to provide offsets if there is a 2d matrix");

  // NOTE: mxfp8 x mxfp8 requires (and asserts later) that offsets is present.
  //       for rowwise, no offsets implies 3d-3d and is handled by lower-level
  //       routines
  if (offs.has_value()) {
    TORCH_CHECK_VALUE(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK_VALUE(offs->dtype() == at::kInt, "Offsets have to be int32");
  }

  const auto out_dtype_ = out_dtype.value_or(kBFloat16);
  TORCH_CHECK_VALUE(out_dtype_ == kBFloat16, "Only bf16 high precision output types are supported for grouped gemm");

  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  // Conversion of implicitly-defined enums to explicit
  auto scale_recipe_a_enum = convert_int_to_enum<ScalingType>(scale_recipe_a);
  auto swizzle_a_enum = convert_int_to_enum<SwizzleType>(swizzle_a);
  auto scale_recipe_b_enum = convert_int_to_enum<ScalingType>(scale_recipe_b);
  auto swizzle_b_enum = convert_int_to_enum<SwizzleType>(swizzle_b);

  // at this point we can start working out what we want to be doing
  // Try to do as few steps as possible.
  // NOTE: support is deliberately sparse, can explicitly enumerate all combinations allowed.
  // Do this via a list of defined (name, acceptance, concrete_impl) tuples.
  ScaledGemmImplementation gemm_impl = ScaledGemmImplementation::NONE;
  for (const auto& fn_entry : scale_grouped_kernel_dispatch) {
    const auto [name, accept_fn, scaled_gemm_impl] = fn_entry;
    bool ok = accept_fn(mat_a.scalar_type(),
                        scale_recipe_a_enum,
                        scale_a,
                        mat_b.scalar_type(),
                        scale_recipe_b_enum,
                        scale_b);
    if (ok) {
      gemm_impl = scaled_gemm_impl;
      break;
    }
  }
  TORCH_CHECK_VALUE(gemm_impl != ScaledGemmImplementation::NONE,
      "No gemm implementation was found");

  switch (gemm_impl) {
    case ScaledGemmImplementation::ROWWISE_ROWWISE: {
      const int scale_multiplier = (mat_a.dim() == 2 && mat_b.dim() == 2) ? offs->size(0) : 1;
      _check_scales_fp8_rowwise(mat_a, scale_a[0], 0 /* dim */ , 0 /* arg_idx */, scale_multiplier);
      _check_scales_fp8_rowwise(mat_b, scale_b[0], 1 /* dim */ , 1 /* arg_idx */, scale_multiplier);
      return _f8_f8_bf16_rowwise_grouped_mm(
          mat_a,
          mat_b,
          scale_a[0],
          scale_b[0],
          offs,
          bias,
          use_fast_accum,
          out);
    }
    case ScaledGemmImplementation::MXFP8_MXFP8: {
      // scale shape checks
      _check_scales_blocked(mat_a, scale_a[0], 0 /* dim */, 0 /* arg_idx */);
      _check_scales_blocked(mat_b, scale_b[0], 1 /* dim */, 1 /* arg_idx */);
      return _mx8_mx8_bf16_grouped_mm_fbgemm(
          mat_a,
          mat_b,
          scale_a[0],
          swizzle_a_enum[0],
          scale_b[0],
          swizzle_b_enum[0],
          offs.value(),
          out);
    }
    case ScaledGemmImplementation::MXFP4_MXFP4: {
      // scale shape checks
      _check_scales_blocked(mat_a, scale_a[0], 0 /* dim */, 0 /* arg_idx */);
      _check_scales_blocked(mat_b, scale_b[0], 1 /* dim */, 1 /* arg_idx */);
      return _f4_f4_bf16_grouped_mm_fbgemm(
          mat_a,
          mat_b,
          scale_a[0], /* block-scale A */
          std::nullopt, /* global-scale A */
          scale_b[0], /* block-scale B */
          std::nullopt, /* global-scale B */
          offs.value(),
          std::nullopt, /* bias */
          out);
    }
    case ScaledGemmImplementation::NVFP4_NVFP4: {
      // scale shape checks
      _check_scales_blocked(mat_a, scale_a[0], 0 /* dim */, 0 /* arg_idx */);
      _check_scales_blocked(mat_b, scale_b[0], 1 /* dim */, 1 /* arg_idx */);
      return _f4_f4_bf16_grouped_mm_fbgemm(
          mat_a,
          mat_b,
          scale_a[0], /* block-scale A */
          scale_a[1], /* global-scale A */
          scale_b[0], /* block-scale B */
          scale_b[1], /* global-scale B */
          offs.value(),
          std::nullopt, /* bias */
          out);
    }
    default:
      TORCH_CHECK_NOT_IMPLEMENTED(false,
          "_scaled_grouped_mm_cuda_v2 is in an inconsistent state - should never reach here");
  }
}

Tensor _grouped_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  bool a_b_and_out_are_bf16 = (
    mat_a.dtype() == at::kBFloat16 &&
    mat_b.dtype() == at::kBFloat16 &&
    out_dtype.value_or(at::kBFloat16) == at::kBFloat16
  );
#ifndef USE_ROCM
  bool use_fast_path = _scaled_mm_allowed_device(/*sm90_only*/true, /*sm100_only*/true) && a_b_and_out_are_bf16;
#else
  // _scaled_mm_allowed_device is used here within _grouped_mm_cuda which seems incorrect since scale is not used.
  // the _grouped_mm_fallback should be safe for any ROCm GPU since it's just calling typical mm/bmm
  bool use_fast_path = false;
  // On non CK system(w/ ROCm), make sure use_fast_path is false
#if defined(USE_ROCM_CK_GEMM)
  if (at::detail::getCUDAHooks().isGPUArch({"gfx942", "gfx950"})) {
    use_fast_path = true;
  }
#endif //USE_ROCM_CK_GEMM
#endif
  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
  if (use_fast_path) {
    // fast path, no d2h sync needed
#ifndef USE_ROCM
    at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
#else
#if defined(USE_ROCM_CK_GEMM)
    at::hip::detail::group_gemm_ck(mat_a, mat_b, offs, bias, out);
#else
    TORCH_WARN("ROCm: Group Gemm through CK not selected.");
#endif //USE_ROCM_CK_GEMM
#endif
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}

} // namespace at::native

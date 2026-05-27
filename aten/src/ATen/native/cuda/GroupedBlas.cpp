#include <cstdint>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/native/ScaledBlasUtils.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/cuda/tunable/TunableGemm.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/cuda/RowwiseScaledMM.h>
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/GroupMM.h>
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
#include <ATen/native/hip/ck_group_gemm.h>
#endif
#include <ATen/ceil_div.h>

#ifdef USE_MSLK
#include <mslk/gemm/gemm_torch.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_addmm_activation_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_foreach_add.h>
#include <ATen/ops/_foreach_mm.h>
#include <ATen/ops/_foreach_mm_native.h>
#include <ATen/ops/_foreach_mul.h>
#include <ATen/ops/_grouped_mm_native.h>
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

namespace scaled_blas = at::native::scaled;
using scaled_blas::ScaledGemmImplementation;
using scaled_blas::convert_int_to_enum;

namespace at::native {

namespace {

bool _scaled_mm_allowed_device(bool sm90_only=false, bool sm100_only=false) {
#ifdef USE_ROCM
  static const std::vector<std::string> archs = {
    "gfx942",
#if ROCM_VERSION >= 60300
    "gfx1200", "gfx1201",
#endif
#if ROCM_VERSION >= 60500
    "gfx950"
#endif
};
  return at::detail::getCUDAHooks().isGPUArch(archs);
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();

  if (sm90_only || sm100_only) {
    return (sm90_only && dprops->major == 9) || (sm100_only && dprops->major == 10);
  } else {
    return dprops->major >= 9 || (dprops->major == 8 && dprops->minor == 9);
  }
#endif
}

// 2d-2d and 2d-3d
// scaling=MXFP8
// CUDA-only
Tensor&
_mx8_mx8_bf16_grouped_mm_mslk(
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

#if defined(USE_MSLK) and !defined(USE_ROCM)
    mslk::gemm::mx8mx8bf16_grouped_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs.value(),
        out);
    return out;
#else
    TORCH_CHECK_NOT_IMPLEMENTED(false, "mxfp8_mxfp8 grouped gemm requires compile with USE_MSLK");
#endif
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
  TORCH_CHECK_VALUE(mat_b.dtype() == at::kFloat8_e4m3fn, "Expected mat_b to be Float8_e4m3 matrix got ", mat_b.scalar_type());

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
#ifdef USE_ROCM
Tensor&
_f8_f8_bf16_rowwise_grouped_mm_rocm(
      const Tensor& mat_a,
      const Tensor& mat_b,
      const Tensor& scale_a,
      const Tensor& scale_b,
      const std::optional<Tensor>& offs,
      Tensor& out) {
  bool is_gfx942 = at::detail::getCUDAHooks().isGPUArch({"gfx942"});

  if (is_gfx942) {
    TORCH_CHECK_VALUE(mat_a.dtype() == at::kFloat8_e4m3fnuz, "Expected mat_a to be Float8_e4m3fnuz matrix got ", mat_a.scalar_type());
    TORCH_CHECK_VALUE(mat_b.dtype() == at::kFloat8_e4m3fnuz, "Expected mat_b to be Float8_e4m3fnuz matrix got ", mat_b.scalar_type());
  } else {
    TORCH_CHECK_VALUE(mat_a.dtype() == at::kFloat8_e4m3fn, "Expected mat_a to be Float8_e4m3 matrix got ", mat_a.scalar_type());
    TORCH_CHECK_VALUE(mat_b.dtype() == at::kFloat8_e4m3fn, "Expected mat_b to be Float8_e4m3 matrix got ", mat_b.scalar_type());
  }

#if defined(USE_MSLK) && defined(USE_ROCM)
  mslk::gemm::f8f8bf16_rowwise_grouped_mm(
      mat_a,
      // FBGEMM expects B matrix shape to be (.., N, K)
      mat_b.transpose(-2, -1),
      scale_a,
      scale_b,
      offs,
      out);
  return out;
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, "grouped gemm is not supported without USE_MSLK on ROCM")
#endif

}
#endif // USE_ROCM

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
_f4_f4_bf16_grouped_mm_mslk(
      const Tensor& mat_a,
      const Tensor& mat_b,
      const Tensor& scale_a,
      const std::optional<Tensor>& global_scale_a,
      const Tensor& scale_b,
      const std::optional<Tensor>& global_scale_b,
      const std::optional<Tensor>& offs,
      const std::optional<Tensor>& bias,
      Tensor& out) {
#if !defined(USE_ROCM) && defined(USE_MSLK)
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

  auto o = mslk::gemm::f4f4bf16_grouped_mm(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      offs.value(),
      out,
      combined_global_scale
  );

  return out;
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, "nvfp4 grouped gemm is not supported without USE_MSLK, and only for CUDA")
#endif
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

    // mslk expects stack of flattened blocked scales for 3d tensor, shape (G, blocked_scale_K * blocked_scale_N).
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

#if defined(USE_MSLK) && defined(USE_CUDA) && !defined(USE_ROCM)
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
    return _mx8_mx8_bf16_grouped_mm_mslk(
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
    if (!contraction_dim.empty()) {
      const int dim_a = contraction_dim[0], dim_b = mat_b.size(contraction_dim[1]);
      TORCH_CHECK_VALUE(mat_a.size(dim_a) == mat_b.size(dim_b),
          "Contraction dimensions (", dim_a, ",", dim_b, ") of mat_a and mat_b must match, got: ", mat_a.size(dim_a), " and ",
          mat_b.size(dim_b));
      // Note: only (-1, -2) is currently supported
      TORCH_CHECK_VALUE(dim_a == -1 && dim_b == -2, "Currently contraction dims must be (-1, -2) only");
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
      // swizze checks
      TORCH_CHECK_VALUE(swizzle_a_enum.size() == 1 && swizzle_b_enum.size() == 1, "Expected single swizzle argument");
      return _mx8_mx8_bf16_grouped_mm_mslk(
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
      return _f4_f4_bf16_grouped_mm_mslk(
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
      return _f4_f4_bf16_grouped_mm_mslk(
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
  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
  if (use_fast_path) {
    // fast path, no d2h sync needed
    at::cuda::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
#else
  // On ROCm fast path routes to group_gemm_ck and slow path to _grouped_mm_fallback.
  // Keep use_fast_path as false till ck kernel perf is optimal.
  // To enable CK path, use env variable ROCM_ALLOW_GROUP_GEMM_CK=1.
  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
#if defined(USE_ROCM_CK_GEMM)
  // ifdef USE_ROCM_CK_GEMM is required since ROCm systems w/o CK should not call ck path.
  // To enable CK path, use env variable ROCM_ALLOW_GROUP_GEMM_CK=1.
  if (at::globalContext().rocmAllowGroupGemmCk() && at::detail::getCUDAHooks().isGPUArch({"gfx942", "gfx950", "gfx90a"})) {
    at::hip::detail::group_gemm_ck(mat_a, mat_b, offs, bias, out);
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
#else
  _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
#endif //USE_ROCM_CK_GEMM
#endif //ifndef USE_ROCM
  return out;
}

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12050
void cublas_foreach_mm(
    at::TensorList self_list,
    at::TensorList mat2_list,
    std::vector<at::Tensor>& outputs) {
  const int group_count = static_cast<int>(self_list.size());
  const auto& first_a = self_list[0];
  const auto& first_b = mat2_list[0];

  bool a_row_major = first_a.stride(-1) == 1;
  bool b_row_major = first_b.stride(-1) == 1;

  // cuBLAS uses column-major layout, so we compute C^T = mat2^T * self^T
  // by passing cuBLAS-A=mat2, cuBLAS-B=self. When a matrix is row-major,
  // cuBLAS already sees it transposed, so op=N; col-major needs op=T.
  cublasOperation_t transa = b_row_major ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t transb = a_row_major ? CUBLAS_OP_N : CUBLAS_OP_T;

  int m = static_cast<int>(first_b.size(1)); // N
  int n = static_cast<int>(first_a.size(0)); // M
  int k = static_cast<int>(first_a.size(1)); // K

  int lda = static_cast<int>(first_b.stride(b_row_major ? 0 : 1));
  int ldb = static_cast<int>(first_a.stride(a_row_major ? 0 : 1));
  int ldc = static_cast<int>(outputs[0].stride(0));

  int group_size = group_count;

  // cublasGemmGroupedBatchedEx requires pointer arrays in DEVICE memory.
  // Build on host in pinned memory, async-copy to device.
  size_t ptrs_bytes = 3 * group_count * sizeof(void*);
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dev_buf = allocator.allocate(ptrs_bytes);
  char* dev_base = static_cast<char*>(dev_buf.get());
  // Layout: A_ptrs[G] | B_ptrs[G] | C_ptrs[G], all as void*
  void** dev_A_ptrs = reinterpret_cast<void**>(dev_base);
  void** dev_B_ptrs = reinterpret_cast<void**>(dev_base + group_count * sizeof(void*));
  void** dev_C_ptrs = reinterpret_cast<void**>(dev_base + 2 * group_count * sizeof(void*));

  auto* host_allocator = at::getHostAllocator(at::kCUDA);
  auto pinned_buf = host_allocator->allocate(ptrs_bytes);
  void** host_ptrs = static_cast<void**>(pinned_buf.get());
  for (int i = 0; i < group_count; i++) {
    host_ptrs[i] = const_cast<void*>(mat2_list[i].data_ptr());
    host_ptrs[group_count + i] = const_cast<void*>(self_list[i].data_ptr());
    host_ptrs[2 * group_count + i] = outputs[i].data_ptr();
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaMemcpyAsync(
      dev_base, host_ptrs, ptrs_bytes,
      cudaMemcpyHostToDevice, stream.stream()));
  host_allocator->record_event(
      pinned_buf.get(), pinned_buf.get_context(), stream.unwrap());

  float alpha = 1.0f;
  float beta = 0.0f;

  cudaDataType_t data_type;
  auto dtype = first_a.scalar_type();
  if (dtype == at::kBFloat16) {
    data_type = CUDA_R_16BF;
  } else if (dtype == at::kHalf) {
    data_type = CUDA_R_16F;
  } else if (dtype == at::kFloat) {
    data_type = CUDA_R_32F;
  } else {
    TORCH_CHECK(false, "cublas_foreach_mm: unsupported dtype ", dtype);
  }

  auto handle = at::cuda::getCurrentCUDABlasHandle();

  TORCH_CUDABLAS_CHECK(cublasGemmGroupedBatchedEx(
      handle,
      &transa,
      &transb,
      &m,
      &n,
      &k,
      &alpha,
      (const void* const*)dev_A_ptrs,
      data_type,
      &lda,
      (const void* const*)dev_B_ptrs,
      data_type,
      &ldb,
      &beta,
      (void* const*)dev_C_ptrs,
      data_type,
      &ldc,
      /*group_count=*/1,
      &group_size,
      CUBLAS_COMPUTE_32F));
}
#endif // !USE_ROCM && CUDA_VERSION >= 12050

#if !defined(USE_ROCM)
// cublasLt grouped GEMM: uses cublasLtGroupedMatrixLayoutCreate (cuBLAS 13.2+)
// Forward-declare the API since it may not be in the compile-time headers.
// Availability is checked at runtime via cublasLtGetVersion().
extern "C" {
cublasStatus_t cublasLtGroupedMatrixLayoutCreate(
    cublasLtMatrixLayout_t* matLayout,
    cudaDataType type,
    int groupCount,
    const void* rows_array,
    const void* cols_array,
    const void* ld_array) __attribute__((weak));
}

// Grouped layout attributes not in old headers
#ifndef CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS
#define CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS ((cublasLtMatmulPreferenceAttributes_t)14)
#define CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS ((cublasLtMatmulPreferenceAttributes_t)15)
#define CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM ((cublasLtMatmulPreferenceAttributes_t)13)
#endif
#ifndef CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE
#define CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE ((cublasLtMatmulDescAttributes_t)39)
#define CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE ((cublasLtMatmulDescAttributes_t)40)
#endif

void cublaslt_foreach_mm(
    at::TensorList self_list,
    at::TensorList mat2_list,
    std::vector<at::Tensor>& outputs) {
  const int group_count = static_cast<int>(self_list.size());
  const auto& first_a = self_list[0];
  const auto& first_b = mat2_list[0];

  bool a_row_major = first_a.stride(-1) == 1;
  bool b_row_major = first_b.stride(-1) == 1;

  // cuBLAS col-major: C^T = mat2^T * self^T => cuBLAS-A=mat2, cuBLAS-B=self
  char transa_c = b_row_major ? 'n' : 't';
  char transb_c = a_row_major ? 'n' : 't';
  cublasOperation_t opa = transa_c == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t opb = transb_c == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;

  int32_t cublas_m = static_cast<int32_t>(first_b.size(1)); // N
  int32_t cublas_n = static_cast<int32_t>(first_a.size(0)); // M
  int32_t cublas_k = static_cast<int32_t>(first_a.size(1)); // K
  int32_t lda = static_cast<int32_t>(first_b.stride(b_row_major ? 0 : 1));
  int32_t ldb = static_cast<int32_t>(first_a.stride(a_row_major ? 0 : 1));
  int32_t ldd = static_cast<int32_t>(outputs[0].stride(0));

  auto dtype = first_a.scalar_type();
  cudaDataType_t cuda_dtype;
  if (dtype == at::kBFloat16) cuda_dtype = CUDA_R_16BF;
  else if (dtype == at::kHalf) cuda_dtype = CUDA_R_16F;
  else if (dtype == at::kFloat) cuda_dtype = CUDA_R_32F;
  else TORCH_CHECK(false, "cublaslt_foreach_mm: unsupported dtype");

  // Device layout: [m(G), n(G), k(G), lda(G), ldb(G), ldd(G)] as int32
  //                [Aptr(G), Bptr(G), Dptr(G), alpha_ptr(G), beta_ptr(G)] as int64
  //                [alpha_scalar, beta_scalar] as float
  const int G = group_count;
  size_t dims_bytes = 6 * G * sizeof(int32_t);
  size_t ptrs_bytes = 5 * G * sizeof(int64_t);
  size_t scalars_bytes = 2 * sizeof(float);
  size_t total = dims_bytes + ptrs_bytes + scalars_bytes;

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dev_buf = allocator.allocate(total);
  char* base = static_cast<char*>(dev_buf.get());

  size_t off = 0;
  int32_t* dev_m = reinterpret_cast<int32_t*>(base + off); off += G * sizeof(int32_t);
  int32_t* dev_n = reinterpret_cast<int32_t*>(base + off); off += G * sizeof(int32_t);
  int32_t* dev_k = reinterpret_cast<int32_t*>(base + off); off += G * sizeof(int32_t);
  int32_t* dev_lda = reinterpret_cast<int32_t*>(base + off); off += G * sizeof(int32_t);
  int32_t* dev_ldb = reinterpret_cast<int32_t*>(base + off); off += G * sizeof(int32_t);
  int32_t* dev_ldd = reinterpret_cast<int32_t*>(base + off); off += G * sizeof(int32_t);
  int64_t* dev_Aptr = reinterpret_cast<int64_t*>(base + off); off += G * sizeof(int64_t);
  int64_t* dev_Bptr = reinterpret_cast<int64_t*>(base + off); off += G * sizeof(int64_t);
  int64_t* dev_Dptr = reinterpret_cast<int64_t*>(base + off); off += G * sizeof(int64_t);
  int64_t* dev_alpha_ptrs = reinterpret_cast<int64_t*>(base + off); off += G * sizeof(int64_t);
  int64_t* dev_beta_ptrs = reinterpret_cast<int64_t*>(base + off); off += G * sizeof(int64_t);
  float* dev_alpha = reinterpret_cast<float*>(base + off); off += sizeof(float);
  float* dev_beta = reinterpret_cast<float*>(base + off);

  // Build all arrays in pinned host memory, async-copy to device in one shot.
  auto* host_allocator = at::getHostAllocator(at::kCUDA);
  auto pinned_buf = host_allocator->allocate(total);
  char* hbase = static_cast<char*>(pinned_buf.get());

  size_t hoff = 0;
  int32_t* h_m = reinterpret_cast<int32_t*>(hbase + hoff); hoff += G * sizeof(int32_t);
  int32_t* h_n = reinterpret_cast<int32_t*>(hbase + hoff); hoff += G * sizeof(int32_t);
  int32_t* h_k = reinterpret_cast<int32_t*>(hbase + hoff); hoff += G * sizeof(int32_t);
  int32_t* h_lda = reinterpret_cast<int32_t*>(hbase + hoff); hoff += G * sizeof(int32_t);
  int32_t* h_ldb = reinterpret_cast<int32_t*>(hbase + hoff); hoff += G * sizeof(int32_t);
  int32_t* h_ldd = reinterpret_cast<int32_t*>(hbase + hoff); hoff += G * sizeof(int32_t);
  int64_t* h_Aptr = reinterpret_cast<int64_t*>(hbase + hoff); hoff += G * sizeof(int64_t);
  int64_t* h_Bptr = reinterpret_cast<int64_t*>(hbase + hoff); hoff += G * sizeof(int64_t);
  int64_t* h_Dptr = reinterpret_cast<int64_t*>(hbase + hoff); hoff += G * sizeof(int64_t);
  int64_t* h_alpha_ptrs = reinterpret_cast<int64_t*>(hbase + hoff); hoff += G * sizeof(int64_t);
  int64_t* h_beta_ptrs = reinterpret_cast<int64_t*>(hbase + hoff); hoff += G * sizeof(int64_t);
  float* h_alpha_scalar = reinterpret_cast<float*>(hbase + hoff); hoff += sizeof(float);
  float* h_beta_scalar = reinterpret_cast<float*>(hbase + hoff);

  *h_alpha_scalar = 1.0f;
  *h_beta_scalar = 0.0f;
  for (int i = 0; i < G; i++) {
    h_m[i] = cublas_m;
    h_n[i] = cublas_n;
    h_k[i] = cublas_k;
    h_lda[i] = lda;
    h_ldb[i] = ldb;
    h_ldd[i] = ldd;
    // cuBLAS-A = mat2, cuBLAS-B = self, cuBLAS-D = output
    h_Aptr[i] = reinterpret_cast<int64_t>(mat2_list[i].data_ptr());
    h_Bptr[i] = reinterpret_cast<int64_t>(self_list[i].data_ptr());
    h_Dptr[i] = reinterpret_cast<int64_t>(outputs[i].data_ptr());
    // alpha/beta pointers point to device scalars (will be valid after copy)
    h_alpha_ptrs[i] = reinterpret_cast<int64_t>(dev_alpha);
    h_beta_ptrs[i] = reinterpret_cast<int64_t>(dev_beta);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaMemcpyAsync(
      base, hbase, total,
      cudaMemcpyHostToDevice, stream.stream()));
  host_allocator->record_event(
      pinned_buf.get(), pinned_buf.get_context(), stream.unwrap());

  // SM 9.0 uses scalar alpha/beta (batch_stride=0), SM 10.0+ uses per-group arrays
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  const bool sm90 = prop->major == 9;
  const int64_t alphaBatchStride = sm90 ? 0 : 1;
  const int64_t betaBatchStride = sm90 ? 0 : 1;
  const auto pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;

  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  // Matmul descriptor
  cublasLtMatmulDesc_t computeDesc;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&computeDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));
  // ALPHA/BETA_BATCH_STRIDE attributes (id 39/40) require cuBLAS 13.2+ headers.
  // On SM 9.0, batch_stride=0 (scalar alpha/beta) is the default, so skip.
  // On SM 10.0+, batch_stride=1 enables per-group alpha/beta pointer arrays.
  if (!sm90) {
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, &alphaBatchStride, sizeof(alphaBatchStride)));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE, &betaBatchStride, sizeof(betaBatchStride)));
  }

  // Grouped matrix layouts
  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
  // A layout: cuBLAS-A = mat2
  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Adesc, cuda_dtype, group_count,
      opa == CUBLAS_OP_N ? static_cast<const void*>(dev_m) : static_cast<const void*>(dev_k),
      opa == CUBLAS_OP_N ? static_cast<const void*>(dev_k) : static_cast<const void*>(dev_m),
      static_cast<const void*>(dev_lda)));
  // B layout: cuBLAS-B = self
  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Bdesc, cuda_dtype, group_count,
      opb == CUBLAS_OP_N ? static_cast<const void*>(dev_k) : static_cast<const void*>(dev_n),
      opb == CUBLAS_OP_N ? static_cast<const void*>(dev_n) : static_cast<const void*>(dev_k),
      static_cast<const void*>(dev_ldb)));
  // C and D layouts: output (m x n in cuBLAS terms)
  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Cdesc, cuda_dtype, group_count,
      static_cast<const void*>(dev_m), static_cast<const void*>(dev_n),
      static_cast<const void*>(dev_ldd)));
  TORCH_CUDABLAS_CHECK(cublasLtGroupedMatrixLayoutCreate(
      &Ddesc, cuda_dtype, group_count,
      static_cast<const void*>(dev_m), static_cast<const void*>(dev_n),
      static_cast<const void*>(dev_ldd)));

  // Heuristic
  cublasLtMatmulPreference_t preference;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
  size_t workspace_size = 32 * 1024 * 1024; // 32MB
  auto workspace = allocator.allocate(workspace_size);
  TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
  // Average dimension hints (cuBLAS 13.2+). Best-effort: ignore if not supported.
  int64_t avgM = cublas_m, avgN = cublas_n, avgK = cublas_k;
  cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, &avgM, sizeof(avgM));
  cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, &avgN, sizeof(avgN));
  cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, &avgK, sizeof(avgK));

  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle, computeDesc, Adesc, Bdesc, Cdesc, Ddesc,
      preference, 1, &heuristicResult, &returnedResult));
  TORCH_CHECK(returnedResult > 0, "cublasLt grouped GEMM: no algorithm found");

  const void* alpha_ptr = sm90 ? static_cast<const void*>(dev_alpha) : static_cast<const void*>(dev_alpha_ptrs);
  const void* beta_ptr = sm90 ? static_cast<const void*>(dev_beta) : static_cast<const void*>(dev_beta_ptrs);

  TORCH_CUDABLAS_CHECK(cublasLtMatmul(
      ltHandle, computeDesc,
      alpha_ptr,
      dev_Aptr, Adesc,
      dev_Bptr, Bdesc,
      beta_ptr,
      dev_Dptr, Cdesc,
      dev_Dptr, Ddesc,
      &heuristicResult.algo,
      workspace.get(), workspace_size,
      stream.stream()));

  // Cleanup descriptors
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatmulDescDestroy(computeDesc);
}
#endif // !USE_ROCM

std::vector<at::Tensor> foreach_tensor_mm_list_kernel_cuda(
    at::TensorList self_list,
    at::TensorList mat2_list) {
  const int64_t group_count = self_list.size();
  TORCH_CHECK(group_count > 0, "_foreach_mm requires non-empty tensor lists");
  TORCH_CHECK(
      group_count == static_cast<int64_t>(mat2_list.size()),
      "_foreach_mm: self and mat2 must have the same number of tensors, got ",
      group_count, " and ", mat2_list.size());

  const auto& first_a = self_list[0];
  const auto& first_b = mat2_list[0];
  TORCH_CHECK(first_a.dim() == 2, "_foreach_mm: tensors in self must be 2D");
  TORCH_CHECK(first_b.dim() == 2, "_foreach_mm: tensors in mat2 must be 2D");

  const int64_t M = first_a.size(0);
  const int64_t K = first_a.size(1);
  const int64_t N = first_b.size(1);
  TORCH_CHECK(
      first_b.size(0) == K,
      "_foreach_mm: contraction dimension mismatch");

  for (int64_t i = 1; i < group_count; i++) {
    TORCH_CHECK(self_list[i].dim() == 2 && mat2_list[i].dim() == 2,
        "_foreach_mm: all tensors must be 2D");
    TORCH_CHECK(
        self_list[i].size(0) == M && self_list[i].size(1) == K,
        "_foreach_mm: all tensors in self must have shape [", M, ", ", K, "]");
    TORCH_CHECK(
        mat2_list[i].size(0) == K && mat2_list[i].size(1) == N,
        "_foreach_mm: all tensors in mat2 must have shape [", K, ", ", N, "]");
    TORCH_CHECK(
        self_list[i].dtype() == first_a.dtype() &&
        mat2_list[i].dtype() == first_b.dtype(),
        "_foreach_mm: all tensors must have the same dtype");
    TORCH_CHECK(
        self_list[i].device() == first_a.device() &&
        mat2_list[i].device() == first_b.device(),
        "_foreach_mm: all tensors must be on the same device");
    TORCH_CHECK(
        self_list[i].stride(0) == first_a.stride(0) &&
        self_list[i].stride(1) == first_a.stride(1),
        "_foreach_mm: all tensors in self must have the same strides");
    TORCH_CHECK(
        mat2_list[i].stride(0) == first_b.stride(0) &&
        mat2_list[i].stride(1) == first_b.stride(1),
        "_foreach_mm: all tensors in mat2 must have the same strides");
  }

  const auto out_dtype = first_a.scalar_type();
  const auto alignment = static_cast<int64_t>(16 / c10::elementSize(out_dtype));
  const int64_t N_padded = (N + alignment - 1) / alignment * alignment;

  std::vector<at::Tensor> outputs;
  outputs.reserve(group_count);
  for (int64_t i = 0; i < group_count; i++) {
    outputs.push_back(at::empty_strided(
        {M, N}, {N_padded, 1}, first_a.options().dtype(out_dtype)));
  }

  bool use_cutlass_path =
      _scaled_mm_allowed_device(/*sm90_only=*/true, /*sm100_only=*/true) &&
      first_a.dtype() == at::kBFloat16 &&
      first_b.dtype() == at::kBFloat16;

  // Allow switching backends via env var for benchmarking.
  // TORCH_FOREACH_MM_CUBLAS=1 → cublasGemmGroupedBatchedEx
  // TORCH_FOREACH_MM_CUBLASLT=1 → cublasLt grouped layout
  const char* cublas_env = std::getenv("TORCH_FOREACH_MM_CUBLAS");
  bool use_cublas = cublas_env != nullptr && cublas_env[0] == '1';
  const char* cublaslt_env = std::getenv("TORCH_FOREACH_MM_CUBLASLT");
  bool use_cublaslt = cublaslt_env != nullptr && cublaslt_env[0] == '1';

#if !defined(USE_ROCM)
  if (use_cublaslt) {
    TORCH_CHECK(cublasLtGroupedMatrixLayoutCreate != nullptr,
        "cublasLt grouped GEMM requires cuBLAS >= 13.2 (CUDA 13.2+). "
        "cublasLtGroupedMatrixLayoutCreate not found in linked library.");
    cublaslt_foreach_mm(self_list, mat2_list, outputs);
  } else
#endif
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12050
  if (use_cublas) {
    cublas_foreach_mm(self_list, mat2_list, outputs);
  } else
#endif
  if (use_cutlass_path) {
    at::cuda::detail::bf16bf16_foreach_mm(self_list, mat2_list, outputs);
  } else {
    for (int64_t i = 0; i < group_count; i++) {
      at::mm_out(outputs[i], self_list[i], mat2_list[i]);
    }
  }

  return outputs;
}

} // namespace at::native

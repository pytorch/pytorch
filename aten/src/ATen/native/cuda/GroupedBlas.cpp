#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>
#include <c10/util/typeid.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/ScaledBlasUtils.h>
#include <ATen/native/cuda/ScaledBlasDeviceUtils.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <ATen/native/GroupedMMUtils.h>
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030
#include <ATen/cuda/detail/CublasLtUtils.h>
#include <ATen/native/cuda/CublasGroupedArgs.h>
#endif
#include <ATen/native/cuda/ScaledGroupMM.h>
#include <ATen/native/cuda/GroupMM.h>
#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
#include <ATen/native/hip/ck_group_gemm.h>
#include <ATen/native/hip/ck_tile_group_gemm.h>
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
#include <ATen/ops/_grouped_mm_native.h>
#include <ATen/ops/_scaled_grouped_mm_native.h>
#include <ATen/ops/_scaled_grouped_mm_v2_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/max.h>
#include <ATen/ops/mm_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/relu.h>
#include <ATen/ops/ones.h>
#endif

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace scaled_blas = at::native::scaled;
using scaled_blas::ScaledGemmImplementation;
using scaled_blas::convert_int_to_enum;
using scaled_blas::scaled_mm_arch_allowed;

namespace at::native {

namespace {

void raise_unsupported_grouped_scaling(
    const Tensor& mat_a,
    const Tensor& mat_b,
    ArrayRef<Tensor> scales_a,
    ArrayRef<ScalingType> recipes_a,
    ArrayRef<SwizzleType> swizzles_a,
    ArrayRef<Tensor> scales_b,
    ArrayRef<ScalingType> recipes_b,
    ArrayRef<SwizzleType> swizzles_b,
    const std::optional<Tensor>& offs,
    ScalarType out_dtype,
    bool use_fast_accum) {
  std::ostringstream inputs;
  inputs << "mat_a: dtype=" << mat_a.scalar_type() << ", shape=" << mat_a.sizes()
         << ", stride=" << mat_a.strides() << ", device=" << mat_a.device()
         << "\nmat_b: dtype=" << mat_b.scalar_type() << ", shape=" << mat_b.sizes()
         << ", stride=" << mat_b.strides() << ", device=" << mat_b.device();
  auto describe_scales = [&](const char* name, ArrayRef<Tensor> scales,
                             ArrayRef<ScalingType> recipes, ArrayRef<SwizzleType> swizzles) {
    inputs << "\n" << name << " recipes=[";
    for (size_t i = 0; i < recipes.size(); ++i) {
      inputs << (i ? ", " : "") << static_cast<int64_t>(recipes[i]);
    }
    inputs << "], swizzles=[";
    for (size_t i = 0; i < swizzles.size(); ++i) {
      inputs << (i ? ", " : "") << static_cast<int64_t>(swizzles[i]);
    }
    inputs << "], scale_count=" << scales.size();
    for (size_t i = 0; i < scales.size(); ++i) {
      inputs << "\n" << name << "[" << i << "]: dtype=" << scales[i].scalar_type()
             << ", shape=" << scales[i].sizes() << ", stride=" << scales[i].strides()
             << ", device=" << scales[i].device();
    }
  };
  describe_scales("scale_a", scales_a, recipes_a, swizzles_a);
  describe_scales("scale_b", scales_b, recipes_b, swizzles_b);
  inputs << "\nout_dtype=" << out_dtype << ", use_fast_accum=" << use_fast_accum;
  if (offs.has_value()) {
    inputs << ", offs: dtype=" << offs->scalar_type() << ", shape=" << offs->sizes()
           << ", stride=" << offs->strides() << ", device=" << offs->device();
  } else {
    inputs << ", offs=None";
  }
  TORCH_CHECK_VALUE(
      false,
      "Invalid scaling configuration for grouped GEMM. Expected:\n"
      "- TensorWise/TensorWise or RowWise/RowWise: float8 inputs, one float32 scale per input; swizzle metadata is ignored.\n"
      "- Hopper block scaling: two float8_e4m3fn inputs, float32 scales packed per group, NO_SWIZZLE; "
      "1x128 is outer-major; 128x128 is K-major with K-block count padded to 4. "
      "3D inputs use (groups, scale_size); 2D inputs concatenate group scales in a 1D tensor. "
      "Recipe pairs: 1x128/1x128, 1x128/128x128, or 128x128/1x128.\n"
      "- Packed MNxK4: matching BlockWise1x32 or BlockWise1x128 recipes, float8_e4m3fn/e5m2 inputs "
      "with at least one e4m3fn input, one float8_e8m0fnu scale per input, SWIZZLE_MNxK4.\n"
      "- MXFP8: matching BlockWise1x32 recipes, float8_e4m3fn inputs, one float8_e8m0fnu scale per input.\n"
      "- MXFP4: matching BlockWise1x32 recipes, packed float4 inputs, one float8_e8m0fnu scale per input.\n"
      "- NVFP4: packed float4 inputs, one BlockWise1x16 float8_e4m3fn scale per input, "
      "optionally followed by one TensorWise float32 global scale on both inputs.\n"
      "cuBLASLt NVFP4 block scales use SWIZZLE_32_4_4; global scales use NO_SWIZZLE. "
      "Omitted NVFP4 swizzles imply the backend layout; MSLK MXFP4/NVFP4 ignore swizzle metadata.\n"
      "Each scale list must match its recipe list.\nGot ", inputs.str());
}

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13040
bool is_hopper_block_impl(ScaledGemmImplementation impl) {
  return impl == ScaledGemmImplementation::BLOCK_1x128_1x128 ||
      impl == ScaledGemmImplementation::BLOCK_1x128_128x128 ||
      impl == ScaledGemmImplementation::BLOCK_128x128_1x128;
}

void _check_cublaslt_grouped_inputs(
    ScaledGemmImplementation impl,
    const Tensor& mat_a,
    const Tensor& mat_b,
    ScalarType out_dtype,
    int64_t batch_count) {
  const auto major = at::cuda::getCurrentDeviceProperties()->major;
  const bool hopper = is_hopper_block_impl(impl);
  if (impl == ScaledGemmImplementation::MNK4_1x32 || impl == ScaledGemmImplementation::MNK4_1x128) {
    at::cuda::blas::detail::check_mnk4_support();
  }
  const bool tensorwise = impl == ScaledGemmImplementation::TENSORWISE_TENSORWISE;
  const bool valid_arch = hopper ? major == 9 : tensorwise
      ? major >= 9 && major <= 11 : major == 10 || major == 11;
  const auto expected_arch = hopper ? "SM9.x" : tensorwise ? "SM9.x/SM10.x/SM11.x" : "SM10.x/SM11.x";
  TORCH_CHECK_NOT_IMPLEMENTED(
      valid_arch, "cuBLASLt grouped GEMM requires ", expected_arch, "; got SM", major, ".x");
  TORCH_CHECK_VALUE(
      batch_count >= 1 && batch_count <= 1024,
      "cuBLASLt grouped GEMM batchCount must be in [1, 1024], got ", batch_count);
  TORCH_CHECK_VALUE(
      out_dtype == kBFloat16 || out_dtype == kHalf || out_dtype == kFloat,
      "cuBLASLt grouped GEMM output dtype must be BFloat16, Float16, or Float32, got ", out_dtype);
  const bool nvfp4 = impl == ScaledGemmImplementation::NVFP4_NVFP4 ||
      impl == ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE;
  if (nvfp4) {
    TORCH_CHECK_VALUE(
        mat_a.scalar_type() == kFloat4_e2m1fn_x2 && mat_b.scalar_type() == kFloat4_e2m1fn_x2,
        "cuBLASLt NVFP4 grouped GEMM inputs must both be float4_e2m1fn_x2; got ",
        mat_a.scalar_type(), " and ", mat_b.scalar_type());
  } else {
    TORCH_CHECK_VALUE(
        scaled_blas::is_valid_cublaslt_input_pair(mat_a.scalar_type(), mat_b.scalar_type()),
        "cuBLASLt grouped GEMM inputs must both be float8_e4m3fn/e5m2 with at least one float8_e4m3fn input; got ",
        mat_a.scalar_type(), " and ", mat_b.scalar_type());
  }
}
#endif

void _check_mslk_grouped_support(
    ScaledGemmImplementation impl,
    const Tensor& mat_a,
    const Tensor& mat_b) {
  const bool mx8 = impl == ScaledGemmImplementation::MXFP8_MXFP8;
  const auto recipe = mx8 ? "MXFP8" : impl == ScaledGemmImplementation::NVFP4_NVFP4 ? "NVFP4" : "MXFP4";
  // MXFP8/NVFP4/MXFP4 use CUDA-only MSLK APIs. Rowwise has a separate ROCm MSLK path.
#if !defined(USE_MSLK)
  TORCH_CHECK_NOT_IMPLEMENTED(false, "MSLK ", recipe, " grouped GEMM requires a build with USE_MSLK");
#endif
#if defined(USE_ROCM)
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "MSLK ", recipe, " grouped GEMM is not supported on ROCm; only rowwise grouped GEMM has a ROCm MSLK path");
#endif
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12080
  TORCH_CHECK_NOT_IMPLEMENTED(false, "MSLK ", recipe, " grouped GEMM requires CUDA >= 12.8");
#endif
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK_NOT_IMPLEMENTED(
      scaled_mm_arch_allowed(/*sm90_only*/ false, /*sm100_only*/ true),
      "MSLK ", recipe, " grouped GEMM requires SM10.x; got SM", dprops->major, ".", dprops->minor);
  if (mx8) {
    TORCH_CHECK_VALUE(
        mat_a.scalar_type() == kFloat8_e4m3fn && mat_b.scalar_type() == kFloat8_e4m3fn,
        "MSLK MXFP8 grouped GEMM requires two float8_e4m3fn inputs, got ",
        mat_a.scalar_type(), " and ", mat_b.scalar_type());
  }
}

void _check_mslk_grouped_kernel_inputs(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<Tensor>& offs,
    const Tensor& out,
    const char* recipe) {
  TORCH_CHECK_VALUE(
      mat_a.dim() == 2 && (mat_b.dim() == 2 || mat_b.dim() == 3),
      "MSLK ", recipe, " grouped GEMM requires 2D/2D or 2D/3D inputs, got ",
      mat_a.dim(), "D/", mat_b.dim(), "D with shapes ", mat_a.sizes(), " and ", mat_b.sizes());
  TORCH_CHECK_VALUE(offs.has_value(), "MSLK ", recipe, " grouped GEMM requires offsets");
  const auto batch_count = offs->size(0);
  // MSLK's mx8mx8bf16_grouped_common.cuh and f4f4bf16_grouped_common.cuh
  // launch one setup thread per group in a single block (CUDA limit: 1024 threads).
  TORCH_CHECK_VALUE(
      batch_count >= 1 && batch_count <= 1024,
      "MSLK ", recipe, " grouped GEMM batchCount must be in [1, 1024], got ", batch_count);
  TORCH_CHECK_VALUE(
      out.scalar_type() == kBFloat16,
      "MSLK ", recipe, " grouped GEMM requires BFloat16 output, got ", out.scalar_type());
}

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030
bool should_use_cublaslt_grouped_gemm(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<Tensor>& offs,
    std::optional<c10::ScalarType> out_dtype) {
  const auto dprops = at::cuda::getCurrentDeviceProperties();
  const bool valid_sm =
      dprops->major >= 9 && dprops->major <= 11;
  if (!valid_sm) {
    return false;
  }

  // The arg-packing kernel launches one thread per group in a single block, so
  // cuBLASLt grouped GEMM only handles [1, 1024] groups. Fall back otherwise.
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  const int64_t batchCount = (a_is_2d || b_is_2d)
      ? (offs.has_value() ? offs->size(0) : 0)
      : mat_a.size(0);
  if (batchCount < 1 || batchCount > 1024) {
    return false;
  }

  const bool fp16_grouped_gemm =
      mat_a.dtype() == at::kHalf && mat_b.dtype() == at::kHalf &&
      out_dtype.value_or(at::kHalf) == at::kHalf;
  if (fp16_grouped_gemm) {
    return true;
  }

  const bool bf16_grouped_gemm =
      mat_a.dtype() == at::kBFloat16 && mat_b.dtype() == at::kBFloat16 &&
      out_dtype.value_or(at::kBFloat16) == at::kBFloat16;
  return bf16_grouped_gemm && at::globalContext().preferCublasltGroupedGemm();
}

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13040
std::pair<CublasGroupedScaleLayout, CublasGroupedScaleLayout>
get_cublaslt_grouped_scale_layouts(
    ScaledGemmImplementation impl,
    const Tensor& scale_a,
    const Tensor& scale_b) {
  using Layout = CublasGroupedScaleLayout;
  switch (impl) {
    case ScaledGemmImplementation::TENSORWISE_TENSORWISE:
      return {
          scale_a.numel() == 1 ? Layout::Scalar : Layout::PerBatchScalar,
          scale_b.numel() == 1 ? Layout::Scalar : Layout::PerBatchScalar};
    case ScaledGemmImplementation::BLOCK_128x128_1x128:
      return {Layout::Block128x128F32, Layout::Vec128F32};
    case ScaledGemmImplementation::BLOCK_1x128_128x128:
      return {Layout::Vec128F32, Layout::Block128x128F32};
    case ScaledGemmImplementation::BLOCK_1x128_1x128:
      return {Layout::Vec128F32, Layout::Vec128F32};
    case ScaledGemmImplementation::NVFP4_NVFP4:
    case ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE:
      return {Layout::Vec16UE4M3, Layout::Vec16UE4M3};
    case ScaledGemmImplementation::MNK4_1x32:
      return {Layout::Vec32MnK4UE8M0, Layout::Vec32MnK4UE8M0};
    case ScaledGemmImplementation::MNK4_1x128:
      return {Layout::Vec128MnK4UE8M0, Layout::Vec128MnK4UE8M0};
    default:
      TORCH_CHECK(false, "unsupported cuBLASLt grouped GEMM implementation: ", static_cast<int>(impl));
  }
}

int resolve_cublaslt_grouped_scale_mode(
    CublasGroupedScaleLayout layout,
    ScalingType scaling,
    const Tensor& scale,
    SwizzleType swizzle,
    bool use_fast_accum) {
  if (layout == CublasGroupedScaleLayout::PerBatchScalar) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F;
  }
  return at::cuda::blas::detail::cublasLtMatmulScaleMode(
      scaling, swizzle, scale.scalar_type(), use_fast_accum);
}

void check_cublaslt_grouped_scale_storage(
    const Tensor& mat,
    const Tensor& scale,
    ScalingType scaling,
    CublasGroupedScaleLayout layout,
    int64_t batchCount,
    bool is_a,
    bool outer_is_jagged,
    const char* name) {
  if (scaling == ScalingType::TensorWise) {
    TORCH_CHECK_VALUE(
        scale.numel() == 1 ||
            (scale.dim() == 1 && scale.numel() == batchCount && scale.is_contiguous()),
        name,
        " tensorwise scale must be a single float32 value or a contiguous 1D float32 tensor with one value per group, got dtype ",
        scale.scalar_type(),
        ", shape ",
        scale.sizes(),
        ", and ",
        scale.numel(),
        " elements for ",
        batchCount,
        " groups");
  } else {
    TORCH_CHECK_VALUE(
        scale.is_contiguous() &&
            reinterpret_cast<uintptr_t>(scale.const_data_ptr()) % 16 == 0,
        name,
        " blockwise scale must be a contiguous tensor with a 16-byte aligned data pointer");
    const int64_t packed_multiplier = mat.scalar_type() == at::kFloat4_e2m1fn_x2 ? 2 : 1;
    const int64_t inner = (is_a ? mat.size(-1) : mat.size(-2)) * packed_multiplier;
    const int64_t outer = is_a ? mat.size(-2) : mat.size(-1);
    if (!outer_is_jagged &&
        cublas_grouped_scale_requires_outer_multiple_of_4(layout)) {
      TORCH_CHECK_VALUE(
          outer % 4 == 0,
          name,
          " requires the per-group outer dimension to be divisible by 4, got ",
          outer);
    }
    const int64_t scale_size = cublas_grouped_scale_size_bytes(
        layout, inner, outer) / scale.element_size();
    if (mat.dim() == 3) {
      TORCH_CHECK_VALUE(
          scale.dim() == 2 &&
          scale.size(0) == batchCount &&
          scale.size(1) == scale_size,
          name,
          " blockwise scale for cuBLASLt grouped GEMM must have shape (",
          batchCount,
          ", ",
          scale_size,
          "), got ",
          scale.sizes());
    } else if (layout == CublasGroupedScaleLayout::Vec128F32 ||
               layout == CublasGroupedScaleLayout::Block128x128F32) {
      TORCH_CHECK_VALUE(
          scale.dim() == 1,
          name,
          " blockwise scale for a 2D Hopper grouped GEMM operand must be a 1D tensor, got ",
          scale.sizes());
    }
  }
}

#endif

bool cublaslt_grouped_mm_use_int64(const Tensor& mat_a, const Tensor& mat_b, const Tensor& out) {
  // cuBLAS grouped GEMM packs per-group m/n/k and lda/ldb/ldd into device
  // arrays whose width is either 32-bit or 64-bit. Switch to 64-bit if any
  // dimension or any stride that ends up as a leading dim could overflow
  // int32_t. Per-group deltas (jagged dim) are bounded by the corresponding
  // total size, so checking sizes is sufficient.
  const int64_t int32_max = std::numeric_limits<int32_t>::max();
  const int64_t k_multiplier =
      mat_a.scalar_type() == at::kFloat4_e2m1fn_x2 ? 2 : 1;
  const auto scaled_value_exceeds_int32 = [int32_max, k_multiplier](int64_t value) {
    return value > int32_max / k_multiplier;
  };
  return mat_a.size(-2) > int32_max || mat_b.size(-1) > int32_max ||
      scaled_value_exceeds_int32(mat_a.size(-1)) ||
      scaled_value_exceeds_int32(mat_b.size(-2)) ||
      scaled_value_exceeds_int32(mat_a.stride(-2)) ||
      scaled_value_exceeds_int32(mat_a.stride(-1)) ||
      scaled_value_exceeds_int32(mat_b.stride(-2)) ||
      scaled_value_exceeds_int32(mat_b.stride(-1)) ||
      out.stride(-2) > int32_max;
}
#endif

// This is NOT an exhaustive list of conditions required for cublasLt scaled_grouped_mm
// Instead, it is the subset of checks needed to determine whether cublasLt is valid for
// the impls it shares with MSLK, assuming checks that apply to both backends already ran
bool should_use_cublaslt_scaled_grouped_gemm(
    ScaledGemmImplementation impl,
    const Tensor& mat_a,
    const Tensor& mat_b,
    ArrayRef<Tensor> scale_a,
    ArrayRef<Tensor> scale_b,
    ScalarType out_dtype,
    int64_t batch_count,
    bool use_fast_accum) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13040
  const auto major = at::cuda::getCurrentDeviceProperties()->major;
  if (major != 10 && major != 11) {
    TORCH_WARN_ONCE("cublasLt NVFP4 scaled grouped GEMM is only supported on sm10x and sm110");
    return false;
  }
  if (batch_count < 1 || batch_count > 1024) {
    TORCH_WARN_ONCE("cuBLASLt grouped GEMM requires 1 <= group count <= 1024");
    return false;
  }
  if (out_dtype != kBFloat16 && out_dtype != kHalf && out_dtype != kFloat) {
    TORCH_WARN_ONCE("cuBLASLt grouped GEMM requires float16, bfloat16, or float32 out dtype");
    return false;
  }
  if (use_fast_accum || scale_a[1].numel() != 1 || scale_b[1].numel() != 1 ||
      !scale_a[1].is_contiguous() || !scale_b[1].is_contiguous()) {
    TORCH_WARN_ONCE("cuBLASLt NVFP4 grouped GEMM with requires global scales to be single elements, and does not support fast_accum");
    return false;
  }
  const auto layout = CublasGroupedScaleLayout::Vec16UE4M3;
  const auto valid_scale_storage = [&](const Tensor& mat, const Tensor& scale, bool is_a) {
    if (!scale.is_contiguous() || reinterpret_cast<uintptr_t>(scale.const_data_ptr()) % 16 != 0) {
      TORCH_WARN_ONCE("cuBLASLt grouped GEMM requires contiguous scales aligned to 16 bytes");
      return false;
    }
    const int64_t inner = (is_a ? mat.size(-1) : mat.size(-2)) * 2;
    const int64_t outer = is_a ? mat.size(-2) : mat.size(-1);
    const int64_t scale_size = cublas_grouped_scale_size_bytes(layout, inner, outer) / scale.element_size();
    if (mat.dim() == 3) {
      if (scale.dim() == 2 && scale.size(0) == batch_count && scale.size(1) == scale_size) {
        return true;
      }
      TORCH_WARN_ONCE("scales are the wrong shape for cuBLASLt grouped GEMM");
      return false;
    }
    // Exact jagged scale sizes and offset values are checked by the device arg-packing kernel.
    return true;
  };
  return valid_scale_storage(mat_a, scale_a[0], true) && valid_scale_storage(mat_b, scale_b[0], false);
#else
  return false;
#endif
}

void _check_scales_blocked(const Tensor& mat, const Tensor& scale, int dim, int arg_idx);

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
    _check_mslk_grouped_kernel_inputs(mat_a, mat_b, offs, out, "MXFP8");
    _check_scales_blocked(mat_a, scale_a, 0, 0);
    _check_scales_blocked(mat_b, scale_b, 1, 1);
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
  const auto recipe = scale_a.scalar_type() == kFloat8_e8m0fnu ? "MXFP4" : "NVFP4";
  _check_mslk_grouped_kernel_inputs(mat_a, mat_b, offs, out, recipe);
  _check_scales_blocked(mat_a, scale_a, 0, 0);
  _check_scales_blocked(mat_b, scale_b, 1, 1);
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

static Tensor grouped_mm_cublaslt(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030
  TORCH_CHECK(
      mat_a.dtype() == at::kBFloat16 || mat_a.dtype() == at::kHalf,
      "cublasLt grouped GEMM requires BFloat16 or Float16 input, got ", mat_a.scalar_type());
  TORCH_CHECK(mat_b.dtype() == mat_a.dtype(),
      "mat_a and mat_b must have the same dtype");
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  const int64_t batchCount64 = (a_is_2d || b_is_2d)
      ? offs->size(0) : mat_a.size(0);
  TORCH_CHECK(batchCount64 > 0 && batchCount64 <= 1024,
      "batchCount must be in [1, 1024], got ", batchCount64);
  const int batchCount = static_cast<int>(batchCount64);

  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  const bool needs_int64 = cublaslt_grouped_mm_use_int64(mat_a, mat_b, out);

  cublasGroupedArgs args(mat_a, mat_b, offs, out, batchCount, needs_int64);
  at::cuda::blas::grouped_gemm(args.transa, args.transb,
                               args.mArray, args.m,
                               args.nArray, args.n,
                               args.kArray, args.k,
                               args.alphaPtrArray, args.alphaScalar,
                               mat_a.scalar_type(),
                               args.APtrArray, args.ldaArray,
                               mat_b.scalar_type(),
                               args.BPtrArray, args.ldbArray,
                               args.betaPtrArray, args.betaScalar, out.scalar_type(),
                               args.DPtrArray, args.lddArray,
                               args.DPtrArray, args.lddArray,
                               args.batchCount, args.use_int64);
  return out;
#else
  TORCH_CHECK(false, "cublasLt grouped GEMM requires CUDA >= 13.3 and is not supported on ROCm. Current build does not meet these requirements.");
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030
}

static void scaled_grouped_mm_cublaslt(
    ScaledGemmImplementation gemm_impl,
    const Tensor& mat_a,
    const Tensor& mat_b,
    ArrayRef<Tensor> scale_a,
    ArrayRef<ScalingType> recipe_a,
    ArrayRef<SwizzleType> swizzle_a,
    ArrayRef<Tensor> scale_b,
    ArrayRef<ScalingType> recipe_b,
    ArrayRef<SwizzleType> swizzle_b,
    const std::optional<Tensor>& offs,
    bool use_fast_accum,
    int64_t batchCount,
    Tensor& out) {
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13040
  _check_cublaslt_grouped_inputs(gemm_impl, mat_a, mat_b, out.scalar_type(), batchCount);
  const auto scaling_a = recipe_a[0];
  const auto scaling_b = recipe_b[0];
  const bool hopper = is_hopper_block_impl(gemm_impl);
  if (hopper) {
    TORCH_CHECK_VALUE(
        (swizzle_a.empty() || (swizzle_a.size() == 1 && swizzle_a[0] == SwizzleType::NO_SWIZZLE)) &&
            (swizzle_b.empty() || (swizzle_b.size() == 1 && swizzle_b[0] == SwizzleType::NO_SWIZZLE)),
        "Hopper grouped block scaling requires swizzle_a/swizzle_b to be empty or NO_SWIZZLE");
  }
  // For non-Hopper recipes, omitted swizzles use NO_SWIZZLE as a placeholder;
  // gemm_impl determines the storage layout.
  const auto swizzle_a_value = swizzle_a.empty() ? SwizzleType::NO_SWIZZLE : swizzle_a[0];
  const auto swizzle_b_value = swizzle_b.empty() ? SwizzleType::NO_SWIZZLE : swizzle_b[0];
  const auto [layout_a, layout_b] = get_cublaslt_grouped_scale_layouts(gemm_impl, scale_a[0], scale_b[0]);
  const bool jagged_m = mat_a.dim() == 2 && mat_b.dim() == 3;
  const bool jagged_n = mat_a.dim() == 3 && mat_b.dim() == 2;
  check_cublaslt_grouped_scale_storage(mat_a, scale_a[0], scaling_a, layout_a, batchCount, true, jagged_m, "scale_a");
  check_cublaslt_grouped_scale_storage(mat_b, scale_b[0], scaling_b, layout_b, batchCount, false, jagged_n, "scale_b");
  const auto scale_mode_a = resolve_cublaslt_grouped_scale_mode(
      layout_a, scaling_a, scale_a[0], swizzle_a_value, use_fast_accum);
  const auto scale_mode_b = resolve_cublaslt_grouped_scale_mode(
      layout_b, scaling_b, scale_b[0], swizzle_b_value, use_fast_accum);
  if (scaling_a == ScalingType::TensorWise && scaling_b == ScalingType::TensorWise) {
    TORCH_CHECK_VALUE(
        layout_a == layout_b,
        "cuBLASLt grouped GEMM requires matching scale layouts: both scales must be single values or both must have one value per group");
  }
  if (mat_a.scalar_type() == at::kFloat4_e2m1fn_x2) {
    TORCH_CHECK(!use_fast_accum, "use_fast_accum is not supported for NVFP4");
  }
  std::optional<Tensor> alpha_scale_a;
  std::optional<Tensor> alpha_scale_b;
  if (gemm_impl == ScaledGemmImplementation::NVFP4_NVFP4) {
    alpha_scale_a = scale_a[1];
    alpha_scale_b = scale_b[1];
    TORCH_CHECK(
        alpha_scale_a->numel() == 1 && alpha_scale_b->numel() == 1 &&
            alpha_scale_a->is_contiguous() && alpha_scale_b->is_contiguous(),
        "NVFP4 tensorwise global scales must be contiguous float32 tensors with one element");
  }

  if (out.numel() == 0) {
    return;
  }
  // A user-provided out tensor keeps its strides, but cuBLAS only receives ldd.
  TORCH_CHECK_VALUE(!check_valid_strides_and_return_transposed(out), "Expected out to be row-major");
  const bool needs_int64 = cublaslt_grouped_mm_use_int64(mat_a, mat_b, out);

  cublasGroupedArgs args(
      mat_a,
      mat_b,
      offs,
      out,
      batchCount,
      needs_int64,
      scale_a[0],
      scale_b[0],
      layout_a,
      layout_b,
      alpha_scale_a,
      alpha_scale_b);
  const at::cuda::blas::GroupedGemmScaleOptions scales{
      args.scale_mata_ptr,
      args.scale_matb_ptr,
      use_fast_accum,
      scale_mode_a,
      scale_mode_b};
  at::cuda::blas::grouped_gemm(
      args.transa, args.transb,
      args.mArray, args.m,
      args.nArray, args.n,
      args.kArray, args.k,
      args.alphaPtrArray, args.alphaScalar,
      mat_a.scalar_type(),
      args.APtrArray, args.ldaArray,
      mat_b.scalar_type(),
      args.BPtrArray, args.ldbArray,
      args.betaPtrArray, args.betaScalar, out.scalar_type(),
      args.DPtrArray, args.lddArray,
      args.DPtrArray, args.lddArray,
      args.batchCount, args.use_int64,
      scales);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "cuBLASLt scaled grouped GEMM requires CUDA >= 13.4 and a non-ROCm build");
#endif // !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13040
}

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
  bool allowed_device = scaled_mm_arch_allowed(/*sm90_only=*/true, /*sm100_only=*/true);
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
        offs,
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

using scaled_blas::ScaleKernelDispatchEntry;

std::array<ScaleKernelDispatchEntry, 11> scale_grouped_kernel_dispatch = {{
  { "tensorwise_tensorwise", scaled_blas::check_tensorwise_recipe, ScaledGemmImplementation::TENSORWISE_TENSORWISE},
  { "rowwise_rowwise", scaled_blas::check_rowwise_recipe, ScaledGemmImplementation::ROWWISE_ROWWISE},
  { "block_1x128_128x128", std::bind_front(scaled_blas::check_deepseek_recipe, ScalingType::BlockWise1x128, ScalingType::BlockWise128x128),
    ScaledGemmImplementation::BLOCK_1x128_128x128},
  { "block_128x128_1x128", std::bind_front(scaled_blas::check_deepseek_recipe, ScalingType::BlockWise128x128, ScalingType::BlockWise1x128),
    ScaledGemmImplementation::BLOCK_128x128_1x128},
  { "block_1x128_1x128", std::bind_front(scaled_blas::check_deepseek_recipe, ScalingType::BlockWise1x128, ScalingType::BlockWise1x128),
    ScaledGemmImplementation::BLOCK_1x128_1x128},
  { "mnk4_1x32", std::bind_front(scaled_blas::check_mnk4_recipe, ScalingType::BlockWise1x32),
    ScaledGemmImplementation::MNK4_1x32},
  { "mnk4_1x128", std::bind_front(scaled_blas::check_mnk4_recipe, ScalingType::BlockWise1x128),
    ScaledGemmImplementation::MNK4_1x128},
  { "mxfp8_mxfp8", scaled_blas::check_mxfp8_recipe, ScaledGemmImplementation::MXFP8_MXFP8},
  { "mxfp4_mxfp4", scaled_blas::check_mxfp4_recipe, ScaledGemmImplementation::MXFP4_MXFP4},
  { "nvfp4_nvfp4", scaled_blas::check_nvfp4_recipe, ScaledGemmImplementation::NVFP4_NVFP4},
  { "nvfp4_nvfp4_single_scale", scaled_blas::check_nvfp4_recipe_single_scale, ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE}}};

} // anonymous namespace

// V2: Grouped scaled matrix multiply. Shape inference + output allocation and
// recipe-independent input validation run in TORCH_META_FUNC(_scaled_grouped_mm_v2);
// this impl handles strides/dtype validation that needs device context, the
// per-recipe scale-shape checks, and kernel dispatch.
TORCH_IMPL_FUNC(_scaled_grouped_mm_cuda_v2_out)(
          const Tensor& mat_a, const Tensor& mat_b,
          const at::ITensorListRef& scale_a_list,
          IntArrayRef scale_recipe_a,
          IntArrayRef swizzle_a,
          const at::ITensorListRef& scale_b_list,
          IntArrayRef scale_recipe_b,
          IntArrayRef swizzle_b,
          at::OptionalTensorRef offs,
          at::OptionalTensorRef bias,
          std::optional<c10::ScalarType> out_dtype,
          IntArrayRef contraction_dim,
          bool use_fast_accum,
          const Tensor& out) {
  TORCH_CHECK_VALUE(!check_valid_strides_and_return_transposed(mat_a), "Expected mat1 to not be transposed");
  TORCH_CHECK_VALUE(check_valid_strides_and_return_transposed(mat_b), "Expected mat2 to be transposed");

  // Materialize the scale lists so the existing acceptance helpers (which take
  // ArrayRef<Tensor>) work unchanged.
  std::vector<Tensor> scale_a(scale_a_list.begin(), scale_a_list.end());
  std::vector<Tensor> scale_b(scale_b_list.begin(), scale_b_list.end());
  ArrayRef<Tensor> scale_a_ref(scale_a);
  ArrayRef<Tensor> scale_b_ref(scale_b);

  // Bridge the optional refs to std::optional<Tensor> for the lower-level kernels.
  std::optional<Tensor> offs_opt =
      offs.has_value() ? std::optional<Tensor>{*offs} : std::nullopt;
  std::optional<Tensor> bias_opt =
      bias.has_value() ? std::optional<Tensor>{*bias} : std::nullopt;

  // The output has already been sized by the structured-op meta function.
  Tensor& out_mut = const_cast<Tensor&>(out);
#ifdef USE_ROCM
  // Preserve the previous `at::zeros` semantics for the 2d-2d case: the CK
  // kernel may not write the whole output region for K=0 / small-K groups.
  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    out_mut.zero_();
  }
#endif

  // Conversion of implicitly-defined enums to explicit
  auto scale_recipe_a_enum = convert_int_to_enum<ScalingType>(scale_recipe_a);
  auto swizzle_a_enum = convert_int_to_enum<SwizzleType>(swizzle_a);
  auto scale_recipe_b_enum = convert_int_to_enum<ScalingType>(scale_recipe_b);
  auto swizzle_b_enum = convert_int_to_enum<SwizzleType>(swizzle_b);

  // at this point we can start working out what we want to be doing
  // Try to do as few steps as possible.
  // NOTE: support is deliberately sparse, can explicitly enumerate all combinations allowed.
  // Do this via a list of defined (name, acceptance, concrete_impl) tuples.
  ScaledGemmImplementation gemm_impl = scaled_blas::find_scaled_gemm_impl(
      scale_grouped_kernel_dispatch,
      mat_a.scalar_type(),
      scale_recipe_a_enum,
      scale_a_ref,
      swizzle_a_enum,
      mat_b.scalar_type(),
      scale_recipe_b_enum,
      scale_b_ref,
      swizzle_b_enum);
  const int64_t batch_count = (mat_a.dim() == 2 || mat_b.dim() == 2)
      ? offs->size(0) : mat_a.size(0);

  switch (gemm_impl) {
    // These cases are only supported by cuBLASLt
    case ScaledGemmImplementation::TENSORWISE_TENSORWISE:
    case ScaledGemmImplementation::BLOCK_1x128_1x128:
    case ScaledGemmImplementation::BLOCK_1x128_128x128:
    case ScaledGemmImplementation::BLOCK_128x128_1x128:
    case ScaledGemmImplementation::MNK4_1x32:
    case ScaledGemmImplementation::MNK4_1x128:
    case ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE: {
      scaled_grouped_mm_cublaslt(
          gemm_impl, mat_a, mat_b, scale_a_ref, scale_recipe_a_enum, swizzle_a_enum,
          scale_b_ref, scale_recipe_b_enum, swizzle_b_enum, offs_opt, use_fast_accum, batch_count, out_mut);
      return;
    }
    // These cases are only supported by MSLK
    case ScaledGemmImplementation::ROWWISE_ROWWISE: {
      TORCH_CHECK_VALUE(
          out.scalar_type() == kBFloat16,
          "Rowwise grouped GEMM requires BFloat16 output, got ", out.scalar_type());
      TORCH_CHECK_NOT_IMPLEMENTED(
          scaled_mm_arch_allowed(/*sm90_only*/ true, /*sm100_only*/ true),
          "Rowwise grouped GEMM is only supported on CUDA devices with "
          "compute capability = [9.0, 10.0], or ROCm MI300+");
      const int scale_multiplier = (mat_a.dim() == 2 && mat_b.dim() == 2) ? offs->size(0) : 1;
      _check_scales_fp8_rowwise(mat_a, scale_a[0], 0 /* dim */ , 0 /* arg_idx */, scale_multiplier);
      _check_scales_fp8_rowwise(mat_b, scale_b[0], 1 /* dim */ , 1 /* arg_idx */, scale_multiplier);
      _f8_f8_bf16_rowwise_grouped_mm(
          mat_a,
          mat_b,
          scale_a[0],
          scale_b[0],
          offs_opt,
          bias_opt,
          use_fast_accum,
          out_mut);
      return;
    }
    case ScaledGemmImplementation::MXFP8_MXFP8: {
      _check_mslk_grouped_support(gemm_impl, mat_a, mat_b);
      // swizze checks
      TORCH_CHECK_VALUE(swizzle_a_enum.size() == 1 && swizzle_b_enum.size() == 1, "Expected single swizzle argument");
      _mx8_mx8_bf16_grouped_mm_mslk(
          mat_a,
          mat_b,
          scale_a[0],
          swizzle_a_enum[0],
          scale_b[0],
          swizzle_b_enum[0],
          offs_opt,
          out_mut);
      return;
    }
    case ScaledGemmImplementation::MXFP4_MXFP4: {
      _check_mslk_grouped_support(gemm_impl, mat_a, mat_b);
      _f4_f4_bf16_grouped_mm_mslk(
          mat_a,
          mat_b,
          scale_a[0], /* block-scale A */
          std::nullopt, /* global-scale A */
          scale_b[0], /* block-scale B */
          std::nullopt, /* global-scale B */
          offs_opt,
          std::nullopt, /* bias */
          out_mut);
      return;
    }
    // These cases are supported by both backends
    case ScaledGemmImplementation::NVFP4_NVFP4: {
#ifdef USE_MSLK
      constexpr bool has_mslk = true;
#else
      constexpr bool has_mslk = false;
#endif
      if (!has_mslk || (at::globalContext().preferCublasltGroupedGemm() &&
          should_use_cublaslt_scaled_grouped_gemm(
              gemm_impl, mat_a, mat_b, scale_a_ref, scale_b_ref, out.scalar_type(), batch_count, use_fast_accum))) {
        scaled_grouped_mm_cublaslt(
            gemm_impl, mat_a, mat_b, scale_a_ref, scale_recipe_a_enum, swizzle_a_enum,
            scale_b_ref, scale_recipe_b_enum, swizzle_b_enum, offs_opt, use_fast_accum, batch_count, out_mut);
        return;
      }
      _check_mslk_grouped_support(gemm_impl, mat_a, mat_b);
      _f4_f4_bf16_grouped_mm_mslk(
          mat_a,
          mat_b,
          scale_a[0], /* block-scale A */
          scale_a[1], /* global-scale A */
          scale_b[0], /* block-scale B */
          scale_b[1], /* global-scale B */
          offs_opt,
          std::nullopt, /* bias */
          out_mut);
      return;
    }
    // Unsupported
    case ScaledGemmImplementation::NONE:
      break;
    default:
      TORCH_CHECK_NOT_IMPLEMENTED(false,
          "_scaled_grouped_mm_cuda_v2 is in an inconsistent state - should never reach here");
  }
  raise_unsupported_grouped_scaling(
      mat_a, mat_b, scale_a_ref, scale_recipe_a_enum, swizzle_a_enum,
      scale_b_ref, scale_recipe_b_enum, swizzle_b_enum, offs_opt, out.scalar_type(), use_fast_accum);
}

#if defined(USE_ROCM) && defined(USE_ROCM_CK_GEMM)
// Inputs group_gemm_ck_tile handles; anything else falls through to the next grouped mm path.
static bool ck_tile_group_gemm_supported(const Tensor& mat_a, const Tensor& mat_b, const std::optional<at::Tensor>& offs) {
  const auto dtype = mat_a.scalar_type();
  if (mat_b.scalar_type() != dtype || (dtype != at::kBFloat16 && dtype != at::kHalf)) { // no dtype mixing yet
    return false;
  }
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  const int64_t group_count = a_is_2d && b_is_2d ? offs->size(0) : (a_is_2d ? mat_b.size(0) : mat_a.size(0));
  if (group_count > 1024) {
    return false;
  }
  // After group_gemm_ck_tile's layout canonicalization, the contiguous dim of A and B must be 16-byte aligned.
  const int64_t M = mat_a.size(-2), K = mat_a.size(-1), N = mat_b.size(-1);
  bool a_row_major = !check_valid_strides_and_return_transposed(mat_a);
  bool b_row_major = !check_valid_strides_and_return_transposed(mat_b);
  if (a_is_2d && b_is_2d) { a_row_major = false; b_row_major = true; }
  else if (a_is_2d) { a_row_major = true; }
  else if (b_is_2d) { a_row_major = true; b_row_major = false; }
  else if (!a_row_major && !b_row_major) { a_row_major = true; }
  const int64_t vector_elems = 16 / mat_a.element_size();
  return (a_row_major ? K : M) % vector_elems == 0 && (b_row_major ? N : K) % vector_elems == 0;
}
#endif

Tensor _grouped_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 13030
  if (should_use_cublaslt_grouped_gemm(mat_a, mat_b, offs, out_dtype)) {
    return grouped_mm_cublaslt(mat_a, mat_b, offs, bias, out_dtype);
  }
#endif
  bool a_b_and_out_are_bf16 = (
    mat_a.dtype() == at::kBFloat16 &&
    mat_b.dtype() == at::kBFloat16 &&
    out_dtype.value_or(at::kBFloat16) == at::kBFloat16
  );
#ifndef USE_ROCM
  bool use_fast_path = scaled_mm_arch_allowed(/*sm90_only=*/true, /*sm100_only=*/true) && a_b_and_out_are_bf16;
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
  // To enable CK-tile path, use env variable ROCM_ALLOW_GROUP_GEMM_CK_TILE=1.
  // To enable CK path, use env variable ROCM_ALLOW_GROUP_GEMM_CK=1.
  if (at::globalContext().rocmAllowGroupGemmCkTile() && at::detail::getCUDAHooks().isGPUArch({"gfx942", "gfx950"})
      && ck_tile_group_gemm_supported(mat_a, mat_b, offs)) {
    at::hip::detail::group_gemm_ck_tile(mat_a, mat_b, offs, bias, out);
  } else if (at::globalContext().rocmAllowGroupGemmCk() && at::detail::getCUDAHooks().isGPUArch({"gfx942", "gfx950", "gfx90a"})) {
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

} // namespace at::native

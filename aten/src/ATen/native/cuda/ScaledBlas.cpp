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
#include <ATen/native/cuda/cuBlasCommonArgs.h>
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

// forward declare
class cublasCommonArgs;

#ifndef _WIN32
namespace fbgemm_gpu {

// NOTE(slayton58): FBGemm_GPU kernels come from <fbgemm_gpu/torch_ops.h> within the FBGemm repo.
//                  To update supported ops means a submodule bump, which is.. painful. Instead, we
//                  can simply forward-declare the methods we want to use.. Works at least as a short-term
//                  thing, but should still be fixed somewhere/somehow.
at::Tensor f4f4bf16(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    bool use_mx);

} // namespace fbgemm_gpu
#endif

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace scaled_blas = at::cuda::scaled;
using scaled_blas::ScaledGemmImplementation;
using scaled_blas::convert_int_to_enum;
using scaled_blas::_scaled_mm_allowed_device;

namespace at::native {

static bool _scaled_mm_allowed_device(bool sm90_only=false, bool sm100_only=false) {
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

#ifdef USE_ROCM
static bool _scaled_mm_is_fnuz() {
    return at::detail::getCUDAHooks().isGPUArch({"gfx942"});
}
#endif

namespace{

/*
 * Scaling Type Determination:
 * ---------------------------
 * Conditions and corresponding Scaling Types:
 *
 * - If scale tensor is `Float8_e8m0fnu` or `Float8_e4m3fn`:
 *   - Returns BlockWise (with additional size checks).
 *
 * - Else if scale.numel() == 1:
 *   - Returns TensorWise.
 *
 * - Else if scale.dim() == 2 && scale.size(0) == outer_dim && scale.size(1) == 1:
 *   - Returns RowWise.
 *
 * - Else if scale.dim() == 2 && scale.size(0) == outer_dim && scale.size(1) == inner_dim / 128:
 *   - Returns BlockWise 1x128.
 *
 * - Else if scale.dim() == 2 && scale.size(0) == outer_dim / 128 && scale.size(1) == inner_dim / 128:
 *   - Returns BlockWise 128x128.
 *
 * - Otherwise:
 *   - Returns Error.
 */

using at::blas::ScalingType;

bool is_tensorwise_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return isFloat8Type(t.scalar_type()) && scale.scalar_type() == kFloat && scale.numel() == 1;
}

bool is_rowwise_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return (isFloat8Type(t.scalar_type()) && scale.scalar_type() == kFloat && scale.dim() == 2
      && scale.size(0) == t.size(0) && scale.size(1) == 1
      && scale.is_contiguous());
}

bool check_size_stride(const at::Tensor& scale, int dim, int size, int stride) {
  // For Blockwise1x128 and Blockwise128x128,
  // when the scale tensor has a dimension of size 1, the stride is effectively
  // "meaningless", i.e. PyTorch decides to use a stride of 1. Thus, the regular
  // stride check fails. Here, we relax the stride check when the effective
  // stride is 1.

  return (
      scale.size(dim) == size && (size <= 1 || scale.stride(dim) == stride));
}

// 1x16 blocks for packed nvfp4 data and fp8_e4m3fn scales
bool is_blockwise_1x16_scaling(const at::Tensor& t, const at::Tensor& scale) {
  // Multiply t.size(1) by 2 to adjust for fp4x2 packing
  // TODO: We might want to enforce some structure on the shapes of the scale
  // tensors
  return (t.scalar_type() == ScalarType::Float4_e2m1fn_x2 && scale.scalar_type() == at::kFloat8_e4m3fn
      && scale.numel() == round_up<int64_t>(t.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(t.size(1) * 2, 16), 4)
      && scale.is_contiguous());
}

// 1x32 blocks for microscaled fp8 data and fp8_e8m0fnu scales
bool is_blockwise_1x32_scaling(const at::Tensor& t, const at::Tensor& scale) {
  // TODO: We might want to enforce some structure on the shapes of the scale
  // tensors
  bool is_fp8_path = (isFloat8Type(t.scalar_type()) && scale.scalar_type() == at::kFloat8_e8m0fnu
      && scale.numel() == round_up<int64_t>(t.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(t.size(1), 32), 4));
  bool is_packed_fp4_path = false;
#ifdef USE_ROCM
  is_packed_fp4_path = (t.scalar_type() == ScalarType::Float4_e2m1fn_x2 && scale.scalar_type() == at::kFloat8_e8m0fnu
      && scale.numel() == round_up<int64_t>(t.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(t.size(1) * 2, 32), 4));
#endif
  return (is_fp8_path || is_packed_fp4_path) && scale.is_contiguous();
}

bool is_blockwise_1x128_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return (
      isFloat8Type(t.scalar_type()) && scale.scalar_type() == kFloat &&
      scale.dim() == 2 && check_size_stride(scale, 0, t.size(0), 1) &&
      check_size_stride(
          scale, 1, ceil_div<int64_t>(t.size(1), 128), t.size(0)));
}

bool is_blockwise_128x128_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return (
      isFloat8Type(t.scalar_type()) && scale.scalar_type() == kFloat &&
      scale.dim() == 2 &&
      check_size_stride(
          scale,
          0,
          ceil_div<int64_t>(t.size(0), 128),
          ceil_div<int64_t>(t.size(1), 128)) &&
      check_size_stride(
          scale, 1, ceil_div<int64_t>(t.size(1), 128), 1));
}

bool is_desired_scaling(const at::Tensor& t, const at::Tensor& scale, ScalingType desired_scaling) {
  switch (desired_scaling) {
    case ScalingType::TensorWise:
      return is_tensorwise_scaling(t, scale);
    case ScalingType::RowWise:
      return is_rowwise_scaling(t, scale);
    case ScalingType::BlockWise1x16:
      return is_blockwise_1x16_scaling(t, scale);
    case ScalingType::BlockWise1x32:
      return is_blockwise_1x32_scaling(t, scale);
    case ScalingType::BlockWise1x128:
      return is_blockwise_1x128_scaling(t, scale);
    case ScalingType::BlockWise128x128:
      return is_blockwise_128x128_scaling(t, scale);
    default:
      TORCH_CHECK(false);
      return false;
  }
}

std::pair<ScalingType, ScalingType> get_joint_scaling(
    std::initializer_list<std::pair<ScalingType, ScalingType>> options,
    const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& scale_a, const at::Tensor& scale_b) {
  for (auto [lhs, rhs] : options) {
    // For blockwise 1x16 and 1x32 scaling, the scale tensors are swizzled/blocked
    // and should not be transposed as their structure is based on the original tensor dimensions
    bool use_swizzled_scale = (rhs == ScalingType::BlockWise1x16 || rhs == ScalingType::BlockWise1x32);
    const at::Tensor& scale_b_check = use_swizzled_scale ? scale_b : scale_b.t();

    if (is_desired_scaling(a, scale_a, lhs) && is_desired_scaling(b.t(), scale_b_check, rhs)) {
      return {lhs, rhs};
    }
  }
  TORCH_CHECK(
    false,
    "Invalid scaling configuration.\n"
    "- For TensorWise scaling, a and b should be float8, scales should be float and singletons.\n"
    "- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (", a.size(0), ", 1) and scale_b should be (1, ", b.size(1), "), and both should be contiguous.\n"
    "- For BlockWise 1x128 scaling, a and b should be float8, scales should be float, scale_a should be (", a.size(0), ", ", ceil_div<int64_t>(a.size(1), 128), ") and scale_b should be (", ceil_div<int64_t>(b.size(0), 128), ", ", b.size(1), "), and both should be outer-dim-major.\n"
    "- For BlockWise 128x128 scaling, a and b should be float8, scales should be float, scale_a should be (", ceil_div<int64_t>(a.size(0), 128), ", ", ceil_div<int64_t>(a.size(1), 128), ") and scale_b should be (", ceil_div<int64_t>(b.size(0), 128), ", ", ceil_div<int64_t>(b.size(1), 128), "), and both should be near-inner-dim-major (with 16-byte aligned strides).\n"
    "- For Blockwise 1x32 scaling, a and b should be float8, scales should be float8_e8m0fnu, scale_a should have ", round_up<int64_t>(a.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(a.size(1), 32), 4), " elements and scale_b should have ", round_up<int64_t>(b.size(1), 128) * round_up<int64_t>(ceil_div<int64_t>(b.size(0), 32), 4), " elements, and both should be contiguous.\n"
    "- For Blockwise 1x16 scaling, a and b should be float4 (packed 2x), scales should be float8_e4m3fn, scale_a should have ", round_up<int64_t>(a.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(a.size(1) * 2, 16), 4), " elements and scale_b should have ", round_up<int64_t>(b.size(1), 128) * round_up<int64_t>(ceil_div<int64_t>(b.size(0) * 2, 16), 4), " elements, and both should be contiguous.\n"
    "Got a.dtype()=", a.scalar_type(), ", scale_a.dtype()=", scale_a.scalar_type(), ", scale_a.size()=", scale_a.sizes(), ", scale_a.stride()=", scale_a.strides(), ", ",
    "b.dtype()=", b.scalar_type(), ", scale_b.dtype()=", scale_b.scalar_type(), ", scale_b.size()=", scale_b.sizes(), " and scale_b.stride()=", scale_b.strides()
  );
}

Tensor&
_tunable_scaled_gemm_rocm(
          cublasCommonArgs& args,
          const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a, const Tensor& scale_b,
          const ScalingType scaling_choice_a, const ScalingType scaling_choice_b,
          const std::optional<Tensor>& bias,
          const bool use_fast_accum,
          const at::ScalarType out_dtype,
          Tensor& out) {
#ifdef USE_ROCM
#define TUNABLE_DISPATCH(BLASOP_A, BLASOP_B)                            \
      if (mat1.scalar_type() == ScalarType::Float8_e4m3fnuz) {        \
        if (mat2.scalar_type() == ScalarType::Float8_e4m3fnuz) {      \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e4m3fnuz, at::Float8_e4m3fnuz, scalar_t,     \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
        else if (mat2.scalar_type() == ScalarType::Float8_e5m2fnuz) { \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e4m3fnuz, at::Float8_e5m2fnuz, scalar_t,     \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
      }                                                               \
      else if (mat1.scalar_type() == ScalarType::Float8_e5m2fnuz) {   \
        if (mat2.scalar_type() == ScalarType::Float8_e4m3fnuz) {      \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e5m2fnuz, at::Float8_e4m3fnuz, scalar_t,     \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
        else if (mat2.scalar_type() == ScalarType::Float8_e5m2fnuz) { \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e5m2fnuz, at::Float8_e5m2fnuz, scalar_t,     \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
      }                                                               \
      else if (mat1.scalar_type() == ScalarType::Float8_e4m3fn) {     \
        if (mat2.scalar_type() == ScalarType::Float8_e4m3fn) {        \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e4m3fn, at::Float8_e4m3fn, scalar_t,         \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
        else if (mat2.scalar_type() == ScalarType::Float8_e5m2) {     \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e4m3fn, at::Float8_e5m2, scalar_t,           \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
      }                                                               \
      else if (mat1.scalar_type() == ScalarType::Float8_e5m2) {       \
        if (mat2.scalar_type() == ScalarType::Float8_e4m3fn) {        \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e5m2, at::Float8_e4m3fn, scalar_t,           \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
        else if (mat2.scalar_type() == ScalarType::Float8_e5m2) {     \
          static at::cuda::tunable::ScaledGemmTunableOp<              \
              at::Float8_e5m2, at::Float8_e5m2, scalar_t,             \
              BLASOP_A, BLASOP_B> scaledgemm{};                       \
          scaledgemm(&params);                                        \
        }                                                             \
      }
  AT_DISPATCH_V2(out_dtype, "_tunable_scaled_gemm", AT_WRAP([&] {
    bool transa_ = ((args.transa != 'n') && (args.transa != 'N'));
    bool transb_ = ((args.transb != 'n') && (args.transb != 'N'));
    at::cuda::tunable::ScaledGemmParams<scalar_t> params;
    params.transa = args.transa;
    params.transb = args.transb;
    params.m = args.m;
    params.n = args.n;
    params.k = args.k;
    params.a = args.mata->data_ptr();
    params.a_scale_ptr = args.scale_mata_ptr;
    params.a_scale_dtype = args.scale_mata_dtype.value();
    params.lda = args.lda;
    params.a_dtype = args.mata->scalar_type();
    params.a_scale_dtype = args.scale_mata_dtype.value();
    params.a_scaling_type = args.scaling_mata_type.value();
    params.b = args.matb->data_ptr();
    params.b_scale_ptr = args.scale_matb_ptr;
    params.b_scale_dtype = args.scale_matb_dtype.value();
    params.ldb = args.ldb;
    params.b_dtype = args.matb->scalar_type();
    params.b_scale_dtype = args.scale_matb_dtype.value();
    params.b_scaling_type = args.scaling_matb_type.value();
    params.bias_ptr = bias ? bias->data_ptr(): nullptr;
    params.bias_dtype = bias ? bias->scalar_type() : isFloat8Type(out_dtype) ? at::ScalarType::Half : out_dtype;
    params.c = args.result->data_ptr();
    params.c_scale_ptr = args.scale_result_ptr;
    params.ldc = args.result_ld;
    params.c_dtype = out_dtype;
    params.use_fast_accum = use_fast_accum;
    if (transa_ && transb_) {
      TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::T)
    }
    else if (transa_ && !transb_) {
      TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::T, at::cuda::tunable::BlasOp::N)
    }
    else if (!transa_ && transb_) {
      TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::T)
    }
    else if (!transa_ && !transb_) {
      TUNABLE_DISPATCH(at::cuda::tunable::BlasOp::N, at::cuda::tunable::BlasOp::N)
    }
    else {
      TORCH_CHECK(false, "unreachable");
    }
  }),
  kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_FLOATING_TYPES));
#undef TUNABLE_DISPATCH
  return out;
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, "_scaled_gemm_rocm only callable on ROCM devices");
#endif
}

Tensor&
_scaled_gemm(
          const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a, const Tensor& scale_b,
          const ScalingType scaling_choice_a, const ScalingType scaling_choice_b,
          const std::optional<Tensor>& bias,
          const bool use_fast_accum,
          Tensor& out,
          const std::optional<Tensor>& alpha = std::nullopt) {
  cublasCommonArgs args(mat1, mat2, out, scale_a, scale_b, std::nullopt, scaling_choice_a, scaling_choice_b);
  const auto out_dtype_ = args.result->scalar_type();
  TORCH_CHECK(args.transa == 't' && args.transb == 'n', "Only multiplication of row-major and column-major matrices is supported by cuBLASLt");

// ROCM enables the TunableOp path only
// but can fallback to at::cuda::blas::scaled_gemm
#ifdef USE_ROCM
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  bool tunable_op_enabled = tuning_ctx->IsTunableOpEnabled();
#else
  bool tunable_op_enabled = false;
#endif
  if (tunable_op_enabled) {
      // Only available on ROCM
      return _tunable_scaled_gemm_rocm(
          args,
          mat1, mat2,
          scale_a, scale_b,
          scaling_choice_a, scaling_choice_b,
          bias,
          use_fast_accum,
          out_dtype_,
          out);
  }
  else
  {
      at::cuda::blas::scaled_gemm(
          args.transa,
          args.transb,
          args.m,
          args.n,
          args.k,
          args.mata->data_ptr(),
          args.scale_mata_ptr,
          args.lda,
          args.mata->scalar_type(),
          args.scale_mata_dtype.value(),
          args.scaling_mata_type.value(),
          args.matb->data_ptr(),
          args.scale_matb_ptr,
          args.ldb,
          args.matb->scalar_type(),
          args.scale_matb_dtype.value(),
          args.scaling_matb_type.value(),
          bias ? bias->data_ptr(): nullptr,
          bias ? bias->scalar_type() : isFloat8Type(out_dtype_) ? at::ScalarType::Half : out_dtype_,
          args.result->data_ptr(),
          args.scale_result_ptr,
          args.result_ld,
          out_dtype_,
          use_fast_accum,
          alpha);
      return out;
  }
}

} // namespace

// NOTE(slayton58): This is defined as part of the _v2 code (way) below - declare the signature here
//                  to help cleanup v1 call structure.
Tensor&
_scaled_rowwise_rowwise(
          const Tensor&, const Tensor&,
          const Tensor&, const Tensor&,
          const std::optional<Tensor>&,
          const c10::ScalarType,
          bool,
          Tensor&);


// Computes matrix multiply + bias while applying scaling to input and output matrices
// Scales are only applicable when matrices are of Float8 type and assumed to be equal to 1.0 by default.
// If output matrix type is 16 or 32-bit type, scale_result is not applied.
// Known limitations:
//  - Only works if mat1 is row-major and mat2 is column-major
//  - Only works if matrices sizes are divisible by 32
//  - If 1-dimensional tensors are used then scale_a should be size = mat1.size(0)
//    and scale_b should have size = to mat2.size(1)
//  Arguments:
//    - `mat1`: the first operand of the matrix multiply, can be type `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `mat2`: the second operand of the matrix multiply, can be type `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `bias`: the bias, can be type `torch.float16` or `torch.bfloat16`
//    - `out_dtype`: the output dtype, can either be a float8 or a higher precision floating point type
//    - `scale_a`: a tensor with the inverse scale of `mat1`, whose shape/strides/dtype depend on the scaling scheme
//    - `scale_b`: a tensor with the inverse scale of `mat2`, whose shape/strides/dtype depend on the scaling scheme
//    - `scale_result`: a scalar tensor with the scale of the output, only utilized if the output is a float8 type
//    - `use_fast_accum`: if true, enables fast float8 accumulation. Backends may ignore this option if not applicable.
//    - `out`: a reference to the output tensor

Tensor&
_scaled_mm_out_cuda(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  // Check sizes
  bool allowed_device = _scaled_mm_allowed_device();
  TORCH_CHECK(allowed_device, "torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+");
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  // Check what type of scaling we are doing based on inputs. This list is sorted
  // by decreasing priority. We prefer "simpler" schemes as they are supported
  // more broadly (more GPU archs, more CUDA versions) and because they are more
  // efficient. This tends to matter only for small matmuls (e.g., 1x1x128).

  // List of supported BlockWise pairs for FP8:
  // https://docs.nvidia.com/cuda/cublas/#element-1d-and-128x128-2d-block-scaling-for-fp8-data-types

  auto [scaling_choice_a, scaling_choice_b] = get_joint_scaling(
    {
      std::make_pair(ScalingType::TensorWise, ScalingType::TensorWise),
      std::make_pair(ScalingType::RowWise, ScalingType::RowWise),
      std::make_pair(ScalingType::BlockWise128x128, ScalingType::BlockWise1x128),
      std::make_pair(ScalingType::BlockWise1x128, ScalingType::BlockWise128x128),
      std::make_pair(ScalingType::BlockWise1x128, ScalingType::BlockWise1x128),
      std::make_pair(ScalingType::BlockWise1x32, ScalingType::BlockWise1x32),
      std::make_pair(ScalingType::BlockWise1x16, ScalingType::BlockWise1x16)
    },
    mat1, mat2, scale_a, scale_b);

  TORCH_CHECK(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
       "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ").");
  TORCH_CHECK(mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0, "mat2 shape (", mat2.sizes()[0], "x",
       mat2.sizes()[1], ") must be divisible by 16");
  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()) || mat1.scalar_type() == ScalarType::Float4_e2m1fn_x2, "Expected mat1 to be Float8 or Float4_x2 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()) || mat2.scalar_type() == ScalarType::Float4_e2m1fn_x2, "Expected mat2 to be Float8 or Float4_x2 matrix got ", mat2.scalar_type());
#ifndef USE_ROCM
  // Type restrictions imposed by CuBLASLt as of CUDA-12.1
  TORCH_CHECK_VALUE(mat1.scalar_type() != ScalarType::Float8_e5m2 || mat2.scalar_type() != ScalarType::Float8_e5m2,
        "Multiplication of two Float8_e5m2 matrices is not supported");
#endif
  if (use_fast_accum) {
    TORCH_CHECK(mat1.scalar_type() != ScalarType::Float4_e2m1fn_x2 && mat2.scalar_type() != ScalarType::Float4_e2m1fn_x2, "`use_fast_accum` is not supported when `mat1` or `mat2` tensors have the `Float4_e2m1fn_x2` dtype.");
  }
#ifdef USE_ROCM
  if (mat1.scalar_type() == ScalarType::Float4_e2m1fn_x2 || mat2.scalar_type() == ScalarType::Float4_e2m1fn_x2) {
    TORCH_CHECK(ROCM_VERSION >= 70000, "Float4_e2m1fn_x2 is only supported for ROCm 7.0 and above");
  }
  if (mat1.scalar_type() == ScalarType::Float8_e5m2 || mat2.scalar_type() == ScalarType::Float8_e5m2) {
    TORCH_CHECK(ROCM_VERSION >= 60500, "Float8_e5m2 is only supported for ROCm 6.5 and above");
  }
  if (mat1.scalar_type() == ScalarType::Float8_e4m3fn || mat2.scalar_type() == ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(ROCM_VERSION >= 60500, "Float8_e4m3fn is only supported for ROCm 6.5 and above");
  }
#endif
  if (bias) {
    TORCH_CHECK(out.scalar_type() != kFloat,
        "Bias is not supported when out_dtype is set to Float32");

    TORCH_CHECK(bias->scalar_type() == ScalarType::BFloat16 ||
                bias->scalar_type() == ScalarType::Half,
        "Bias must be BFloat16 or Half, but got ", bias->scalar_type());

    TORCH_CHECK((out.scalar_type() != kFloat &&
                 out.scalar_type() != ScalarType::BFloat16) ||
                bias->scalar_type() == ScalarType::BFloat16,
        "Bias must be BFloat16 to compute ", out.scalar_type(),
        " output, but got ", bias->scalar_type());

    TORCH_CHECK(out.scalar_type() != ScalarType::Half ||
                bias->scalar_type() == ScalarType::Half,
        "Bias must be Float16 to compute ", out.scalar_type(),
        " output, but got ", bias->scalar_type());
  }
  {
    auto bias_ = bias.value_or(Tensor());
    auto scale_result_ = scale_result.value_or(Tensor());

    // NOLINTNEXTLINE(*c-array*)
    TensorArg targs[]{{out, "out", 0}, {mat1, "mat1", 1}, {mat2, "mat2", 2},
                      {bias_, "bias", 3}, {scale_a, "scale_a", 4}, {scale_b, "scale_b", 5},
                      {scale_result_, "scale_result", 6}};
    checkAllSameGPU(__func__, targs);
  }
  // Validation checks have passed lets resize the output to actual size
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  // If any of M, K, N is 0 - return early (the tensorwise/rowwise float8 gemm kernels
  // do not support this case).
  if (mat1_sizes[0] == 0 || mat1_sizes[1] == 0 || mat2_sizes[1] == 0) {
    // `out` was created with `at::empty`. In the case where we are multiplying
    // MxK by KxN and K is the zero dim, we need to initialize here to properly
    // return a tensor of zeros.
    if (mat1_sizes[1] == 0) {
      out.zero_();
    }

    return out;
  }

  // NVIDIA's cuBLAS only started supporting row-wise scaling in version 12.9,
  // and only for compute capability 9.0+. In other cases we use CUTLASS.
  // We are doing row-wise scaling
  if (scaling_choice_a == ScalingType::RowWise && scaling_choice_b == ScalingType::RowWise) {
#ifndef USE_ROCM
    auto dprops = at::cuda::getCurrentDeviceProperties();
    if ((dprops->major < 9 || CUBLAS_VERSION < 120900 || cublasLtGetVersion() < 120900)
        // cuBLAS only supports tiled 1D factor layout for 1D block scaling, no 2D block scales
        ||  (dprops->major >= 10 && (!scale_a.sizes().empty() || !scale_b.sizes().empty()))) {
      TORCH_CHECK_VALUE(out.dtype() == kBFloat16 || out.dtype() == kHalf, "Only bf16 and fp16 high precision output types are supported for row-wise scaling.");
      return _scaled_rowwise_rowwise(
          mat1,
          mat2,
          scale_a,
          scale_b,
          bias,
          out.scalar_type(),
          use_fast_accum,
          out);
    }
#else
    // For ROCm, match behavior of f8f8bf16_rowwise type checking, for unit test purposes.
    Tensor b = mat2;
    if (_scaled_mm_is_fnuz()) {
      TORCH_CHECK_VALUE(b.dtype() == at::kFloat8_e4m3fnuz,
          "Expected b.dtype() == at::kFloat8_e4m3fnuz, got: ", b.dtype());
    }
    else {
      TORCH_CHECK_VALUE(b.dtype() == at::kFloat8_e4m3fn,
          "Expected b.dtype() == at::kFloat8_e4m3fn, got: ", b.dtype());
    }
    // Until more than bf16 is supported.
    TORCH_CHECK_VALUE(out.scalar_type() == ScalarType::BFloat16,
         "hipblaslt rowwise _scaled_mm only supports BFloat16 output but got ", out.scalar_type());
#endif
  }
  else if (scaling_choice_a == ScalingType::BlockWise1x32 && scaling_choice_b == ScalingType::BlockWise1x32) {
#ifdef USE_ROCM
    #if ROCM_VERSION >= 70000
    TORCH_CHECK_NOT_IMPLEMENTED(at::detail::getCUDAHooks().isGPUArch({"gfx950"}),
                "Block-wise scaling for Float8_e8m0fnu is only supported on gfx950");

    int packed_factor = 1;
    if (mat1.scalar_type() == ScalarType::Float4_e2m1fn_x2) {
      // For float4 data type, each byte stores two 4-bit floating-point values,
      // effectively packing two elements into one byte.
      packed_factor = 2;
    }
    TORCH_CHECK_VALUE(mat1.size(0) % 16 == 0 && (mat1.size(1) * packed_factor) % 128 == 0 &&
                mat2.size(1) % 16 == 0,
                "M, N must be multiples of 16 and K must be multiple of 128 for block-wise scaling");

    TORCH_CHECK_VALUE(out.scalar_type() == ScalarType::BFloat16 ||
                out.scalar_type() == ScalarType::Half,
                "Block-wise scaling only supports BFloat16 or Half output types");
#else
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Block-wise scaling for Float8_e8m0fnu requires ROCm 7.0 or later");
#endif
#endif
  }

  return _scaled_gemm(mat1, mat2, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, use_fast_accum, out);
}

Tensor
_scaled_mm_cuda(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));

  return _scaled_mm_out_cuda(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

using acceptance_fn = std::function<bool(c10::ScalarType, std::vector<ScalingType>&, ArrayRef<Tensor>&, c10::ScalarType, std::vector<ScalingType>&, ArrayRef<Tensor>&)>;
using namespace std::placeholders;

namespace scaled_blas = at::cuda::scaled;
using scaled_blas::ScaledGemmImplementation;
using scaled_blas::convert_int_to_enum;

std::array<std::tuple<std::string, acceptance_fn, ScaledGemmImplementation>, 9> scale_kernel_dispatch = {{
  { "tensorwise_tensorwise", scaled_blas::check_tensorwise_recipe, ScaledGemmImplementation::TENSORWISE_TENSORWISE },
  { "rowwise_rowwise", scaled_blas::check_rowwise_recipe, ScaledGemmImplementation::ROWWISE_ROWWISE},
  { "block_1x128_128x128", std::bind(scaled_blas::check_deepseek_recipe, ScalingType::BlockWise1x128, ScalingType::BlockWise128x128, _1, _2, _3, _4, _5, _6),
    ScaledGemmImplementation::BLOCK_1x128_128x128},
  { "block_128x128_1x128", std::bind(scaled_blas::check_deepseek_recipe, ScalingType::BlockWise128x128, ScalingType::BlockWise1x128, _1, _2, _3, _4, _5, _6),
    ScaledGemmImplementation::BLOCK_128x128_1x128},
  { "block_1x128_1x128", std::bind(scaled_blas::check_deepseek_recipe, ScalingType::BlockWise1x128, ScalingType::BlockWise1x128, _1, _2, _3, _4, _5, _6),
    ScaledGemmImplementation::BLOCK_1x128_1x128},
  { "nvfp4_nvfp4", scaled_blas::check_nvfp4_recipe, ScaledGemmImplementation::NVFP4_NVFP4},
  { "nvfp4_nvfp4_single_scale", scaled_blas::check_nvfp4_recipe_single_scale, ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE },
  { "mxfp8_mxfp8", scaled_blas::check_mxfp8_recipe, ScaledGemmImplementation::MXFP8_MXFP8},
  { "mxfp4_mxfp4", scaled_blas::check_mxfp4_recipe, ScaledGemmImplementation::MXFP4_MXFP4}}};

Tensor&
_scaled_tensorwise_tensorwise(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const Tensor& scale_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  // Restrictions:
  // A, B are FP8, scales are fp32
  //
  TORCH_CHECK_VALUE(isFloat8Type(mat_a.scalar_type()) && isFloat8Type(mat_b.scalar_type()), "mat_a and mat_b must be fp8 types, got: ",
      mat_a.scalar_type(), mat_b.scalar_type());
  TORCH_CHECK_VALUE(scale_a.numel() == 1 && scale_a.scalar_type() == kFloat, "scale_a must have 1 Float element")
  TORCH_CHECK_VALUE(scale_b.numel() == 1 && scale_b.scalar_type() == kFloat, "scale_b must have 1 Float element")

  auto scaling_choice_a = ScalingType::TensorWise;
  auto scaling_choice_b = ScalingType::TensorWise;

  _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, use_fast_accum, out);

  return out;
}


Tensor&
_scaled_rowwise_rowwise(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const Tensor& scale_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  // Restrictions:
  // A, B are FP8, scales are fp32, shape M/N for A/B
  TORCH_CHECK_VALUE(isFloat8Type(mat_a.scalar_type()) && isFloat8Type(mat_b.scalar_type()), "mat_a and mat_b must be fp8 types, got: ",
      mat_a.scalar_type(), mat_b.scalar_type());
  TORCH_CHECK_VALUE(scale_a.size(0) == mat_a.size(0) && scale_a.size(1) == 1, "scale_a must have shape [", mat_a.size(0), ", 1], got [", scale_a.sizes(), "]");
  TORCH_CHECK_VALUE(scale_a.numel() == mat_a.size(0) && scale_a.scalar_type() == kFloat, "scale_a must have ", mat_a.size(0), " Float elements, got ", scale_a.numel())
  TORCH_CHECK_VALUE(scale_b.numel() == mat_b.size(1) && scale_b.scalar_type() == kFloat, "scale_b must have ", mat_b.size(1), " Float elements, got ", scale_b.numel())

  // if we have a scale of shape [256, 1] (say), then stride can be [1, 0] - handle this case
  TORCH_CHECK_VALUE(
      scale_a.stride(1) == 1 ||
      scale_a.size(1) == 1,
      "expected scale_a.stride(1) to be 1, but got ", scale_a.stride(1)
  );
  TORCH_CHECK_VALUE(scale_b.stride(1) == 1, "expected scale_b.stride(1) to be 1, but got ", scale_b.stride(1));

  auto scaling_choice_a = ScalingType::RowWise;
  auto scaling_choice_b = ScalingType::RowWise;
  //
  // NVIDIA's cuBLAS only started supporting row-wise scaling in version 12.9,
  // and only for compute capability 9.0+. In other cases we use CUTLASS.
#ifndef USE_ROCM
  // We are doing row-wise scaling
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (((dprops->major < 9 || CUBLAS_VERSION < 120900 || cublasLtGetVersion() < 120900)
      // cuBLAS only supports tiled 1D factor layout for 1D block scaling, no 2D block scales
      ||  (dprops->major == 10 && (scale_a.sizes().size() || scale_b.sizes().size())))) {
    TORCH_CHECK_VALUE(out.dtype() == kBFloat16 || out.dtype() == kHalf, "Only bf16 and fp16 high precision output types are supported for row-wise scaling.");
    at::cuda::detail::f8f8bf16_rowwise(
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        bias,
        use_fast_accum,
        out);
    return out;
  }
#else

  // For ROCm, match behavior of f8f8bf16_rowwise type checking, for unit test purposes.
  //Tensor b = mat_b;
  if (_scaled_mm_is_fnuz()) {
    TORCH_CHECK_VALUE(mat_b.dtype() == at::kFloat8_e4m3fnuz, "expected mat_b.dtype() to be at::kFloat8_e4m3fnuz, but got ", mat_b.dtype());
  }
  else {
    TORCH_CHECK_VALUE(mat_b.dtype() == at::kFloat8_e4m3fn, "expected mat_b.dtype() to be at::kFloat8_e4m3fn, but got ", mat_b.dtype());
  }
  // Until more than bf16 is supported.
  TORCH_CHECK_VALUE(out.scalar_type() == ScalarType::BFloat16,
       "hipblaslt rowwise _scaled_mm only supports BFloat16 output but got ", out.scalar_type());
#endif

  _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, use_fast_accum, out);

  return out;
}

void
_check_deepseek_support() {
#ifndef USE_ROCM
  auto dprops = at::cuda::getCurrentDeviceProperties();
  if (dprops->major != 9) {
    // Only on Hopper GPUs
    TORCH_CHECK_NOT_IMPLEMENTED(
      dprops->major == 9,
      "DeepSeek style (1x128, 128x128) scaling only supported in CUDA for SM90")
  }
  // Only in cublasLt >= 12.9
  TORCH_CHECK_NOT_IMPLEMENTED(
    CUBLAS_VERSION >= 120900 && cublasLtGetVersion() >= 120900,
    "DeepSeek style (1x128, 128x128) scaling requires cublasLt >= 12.9"
  );
#endif
}

Tensor&
_scaled_block1x128_block1x128(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const Tensor& scale_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          const bool use_fast_accum,
          Tensor& out) {
#ifndef USE_ROCM
  // Restrictions:
  // A, B are FP8, scales are fp32, shape K//128
  // As: [M x K // 128], stride: [1, M]
  // Bs: [N x K // 128], stride: [1, N]
  _check_deepseek_support();

  // check types
  TORCH_CHECK_VALUE(
    isFloat8Type(mat_a.scalar_type()) &&
    isFloat8Type(mat_b.scalar_type()),
    "mat_a and mat_b must be fp8 types, got: ", mat_a.scalar_type(), mat_b.scalar_type()
  );

  const int64_t M = mat_a.sizes()[0];
  const int64_t K = mat_a.sizes()[1];
  const int64_t N = mat_b.sizes()[1];

  // scale_a shape
  TORCH_CHECK_VALUE(
    scale_a.size(0) == M &&
    scale_a.size(1) == ceil_div<int64_t>(K, 128) &&
    scale_a.scalar_type() == kFloat,
    "scale_a must have shape ", M, " x ", ceil_div<int64_t>(K, 128), " Float elements, got ", scale_a.sizes()
  );
  // scale_a stride
  TORCH_CHECK_VALUE(
    scale_a.stride(0) == 1 &&
    (
      scale_a.stride(1) == M ||
      (scale_a.size(1) == 1 && scale_b.stride(1) == 1)
    ),
    "scale_a strides must be (", 1, ", ", M, "); got: ", scale_a.strides()
  );

  // scale_b shape
  TORCH_CHECK_VALUE(
    scale_b.size(0) == N &&
    scale_b.size(1) == ceil_div<int64_t>(K, 128) &&
    scale_b.scalar_type() == kFloat,
    "scale_b must have shape ", N, " x ", ceil_div<int64_t>(K, 128), " Float elements, got ", scale_b.sizes()
  );
  // scale_b stride
  TORCH_CHECK_VALUE(
    scale_b.stride(0) == 1 &&
    (
      scale_b.stride(1) == N ||
      (
        scale_b.size(1) == 1 &&
        scale_b.stride(1) == 1
      )
    ),
    "scale_b strides must be (", 1, ", ", N, "); got: ", scale_a.strides()
  );

  auto scaling_choice_a = ScalingType::BlockWise1x128;
  auto scaling_choice_b = ScalingType::BlockWise1x128;

  _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, use_fast_accum, out);

  return out;
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
    false,
    "1x128 and 128x128 scaling not available with ROCm"
  );
#endif
}

Tensor&
_scaled_block128x128_block1x128(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const Tensor& scale_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          const bool use_fast_accum,
          Tensor& out) {
#ifndef USE_ROCM
  // Restrictions:
  _check_deepseek_support();

  // A: [M, K], B: [K, N] are FP8, scales are fp32
  // As: [round_up(K // 128, 4), M // 128], stride: [M // 128, 1]
  // Bs: [N x K // 128], stride: [1, N]
  TORCH_CHECK_VALUE(
    isFloat8Type(mat_a.scalar_type()) &&
    isFloat8Type(mat_b.scalar_type()),
    "mat_a and mat_b must be fp8 types, got: ",  mat_a.scalar_type(), mat_b.scalar_type()
  );

  const int64_t M = mat_a.sizes()[0];
  const int64_t K = mat_a.sizes()[1];
  const int64_t N = mat_b.sizes()[1];

  // scale_a shape
  TORCH_CHECK_VALUE(
    scale_a.size(0) == round_up<int64_t>(ceil_div<int64_t>(K, 128), 4) &&
    scale_a.size(1) == ceil_div<int64_t>(M, 128) &&
    scale_a.scalar_type() == kFloat,
    "scale_a must have shape ", round_up<int64_t>(ceil_div<int64_t>(K, 128), 4), " x ",
      ceil_div<int64_t>(M, 128), " Float elements, got ", scale_a.sizes()
  );
  // scale_a stride
  TORCH_CHECK_VALUE(
    scale_a.stride(0) == 1 &&
    (
      scale_a.stride(1) == round_up<int64_t>(ceil_div<int64_t>(K, 128), 4) ||
      (
        scale_a.size(1) == 1 &&
        scale_a.stride(1) == 1
      )
    ),
    "scale_a must have strides (1, ", round_up<int64_t>(ceil_div<int64_t>(K, 128), 4), "); got ", scale_b.strides()
  );

  // scale_b shape
  TORCH_CHECK_VALUE(
    scale_b.size(0) == N &&
    scale_b.size(1) == ceil_div<int64_t>(K, 128) &&
    scale_b.scalar_type() == kFloat,
    "scale_b must have shape ", N, " x ", ceil_div<int64_t>(K, 128), " Float elements, got ", scale_b.sizes()
  );
  // scale_b stride
  TORCH_CHECK_VALUE(
    scale_b.stride(0) == 1 &&
    (
      scale_b.stride(1) == N ||
      (
        scale_b.size(1) == 1 &&
        scale_b.stride(1) == 1
      )
    ),
    "scale_b must have strides (1, ", N, "); got ", scale_b.strides()
  );

  auto scaling_choice_a = ScalingType::BlockWise128x128;
  auto scaling_choice_b = ScalingType::BlockWise1x128;

  _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, use_fast_accum, out);

  return out;
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
    false,
    "1x128 and 128x128 scaling not available with ROCm"
  );
#endif
}

Tensor&
_scaled_block1x128_block128x128(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const Tensor& scale_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          const bool use_fast_accum,
          Tensor& out) {
#ifndef USE_ROCM
  // Restrictions:
  _check_deepseek_support();
  // A: [M, K], B: [K, N] are FP8, scales are fp32
  // As: [M x K // 128], stride: [1, M]
  // Bs: [round_up(K // 128, 4) x N // 128], stride: [1, N // 128]
  TORCH_CHECK_VALUE(
    isFloat8Type(mat_a.scalar_type()) &&
    isFloat8Type(mat_b.scalar_type()),
    "mat_a and mat_b must be fp8 types, got: ", mat_a.scalar_type(), mat_b.scalar_type()
  );

  int64_t M = mat_a.size(0);
  int64_t K = mat_a.size(1);
  int64_t N = mat_b.size(1);

  // scale_a shape
  TORCH_CHECK_VALUE(
    scale_a.size(0) == M &&
    scale_a.size(1) == ceil_div<int64_t>(K, 128) &&
    scale_a.scalar_type() == kFloat,
    "scale_a must have shape ", M, " x ", ceil_div<int64_t>(K, 128), " Float elements, got ", scale_a.sizes()
  );
  // scale_a stride
  TORCH_CHECK_VALUE(
    scale_a.stride(0) == 1 &&
    (
      scale_a.stride(1) == M ||
      (
        scale_a.size(1) == 1 &&
        scale_a.stride(1) == 1
      )
    ),
    "scale_a must have strides (1, ", M, "); got ", scale_b.strides()
  );
  // scale_b shape
  TORCH_CHECK_VALUE(
    scale_b.size(0) == round_up<int64_t>(ceil_div<int64_t>(K, 128), 4) &&
    scale_b.size(1) == ceil_div<int64_t>(N, 128) &&
    scale_b.scalar_type() == kFloat,
    "scale_b must have shape ", round_up<int64_t>(ceil_div<int64_t>(K, 128), 4), " x ", ceil_div<int64_t>(N, 128), " Float elements, got ", scale_b.sizes()
  );
  // scale_b stride
  TORCH_CHECK_VALUE(
    scale_b.stride(0) == 1 &&
    (
      scale_b.stride(1) == round_up<int64_t>(ceil_div<int64_t>(K, 128), 4) ||
      (
        scale_b.size(1) == 1 &&
        scale_b.stride(1) == 1
      )
    ),
    "scale_b must have strides (1, ", round_up<int64_t>(ceil_div<int64_t>(K, 128), 4), "); got ", scale_b.strides()
  );

  auto scaling_choice_a = ScalingType::BlockWise1x128;
  auto scaling_choice_b = ScalingType::BlockWise128x128;

  _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, use_fast_accum, out);

  return out;
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
    false,
    "1x128 and 128x128 scaling not available with ROCm"
  );
#endif
}

Tensor&
_scaled_mxfp8_mxfp8(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const SwizzleType swizzle_a,
          const Tensor& scale_b, const SwizzleType swizzle_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          Tensor& out) {
  // Restrictions:
  // A, B are FP8, scales are e8m0, A: shape K//32, B: K, N//32
  // Scales must be swizzled
  TORCH_CHECK_VALUE(isFloat8Type(mat_a.scalar_type()) && isFloat8Type(mat_b.scalar_type()), "mat_a and mat_b must be fp8 types, got: ",
      mat_a.scalar_type(), mat_b.scalar_type());

#ifdef USE_ROCM
  auto scale_a_elems = ceil_div<int64_t>(mat_a.size(0), 32) * mat_a.size(1);
  auto scale_b_elems = ceil_div<int64_t>(mat_b.size(1), 32) * mat_b.size(0);
#else
  auto scale_a_elems = round_up<int64_t>(mat_a.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_a.size(1), 32), 4);
  auto scale_b_elems = round_up<int64_t>(mat_b.size(1), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_b.size(0), 32), 4);
#endif
  TORCH_CHECK_VALUE(scale_a_elems == scale_a.numel(),
         "For Blockwise scaling scale_a should have ", scale_a_elems, " elements, got: ", scale_a.numel());
  TORCH_CHECK_VALUE(scale_b_elems == scale_b.numel(),
         "For Blockwise scaling scale_b should have ", scale_b_elems, " elements, got: ", scale_b.numel());

#ifndef USE_ROCM
  TORCH_CHECK_VALUE(swizzle_a == SwizzleType::SWIZZLE_32_4_4, "scale_a must be swizzled to SWIZZLE_32_4_4 format");
  TORCH_CHECK_VALUE(swizzle_b == SwizzleType::SWIZZLE_32_4_4, "scale_b must be swizzled to SWIZZLE_32_4_4 format");
#endif

  TORCH_CHECK_VALUE(scale_a.is_contiguous() && scale_b.is_contiguous(),
        "For Blockwise scaling both scales should be contiguous");

  TORCH_CHECK_VALUE(out.scalar_type() == out_dtype, "expected out.scalar_type() to be ", out_dtype, ", but got ", out_dtype);

  auto scaling_choice_a = ScalingType::BlockWise1x32;
  auto scaling_choice_b = ScalingType::BlockWise1x32;

#ifdef USE_ROCM
#if ROCM_VERSION >= 70000
  TORCH_CHECK_NOT_IMPLEMENTED(at::detail::getCUDAHooks().isGPUArch({"gfx950"}),
              "Block-wise scaling for Float8_e8m0fnu is only supported on gfx950");

  TORCH_CHECK_VALUE(mat_a.size(0) % 32 == 0 && mat_a.size(1) % 32 == 0 &&
              mat_b.size(0) % 32 == 0 && mat_b.size(1) % 32 == 0,
              "Matrix dimensions must be multiples of 32 for block-wise scaling");

  TORCH_CHECK_VALUE(out.scalar_type() == ScalarType::BFloat16 ||
              out.scalar_type() == ScalarType::Half,
              "Block-wise scaling only supports BFloat16 or Half output types");
#else
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Block-wise scaling for Float8_e8m0fnu requires ROCm 7.0 or later");
#endif
#endif

  return _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, false /* use_fast_accum */, out);
}


Tensor&
_scaled_mxfp4_mxfp4(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const SwizzleType swizzle_a,
          const Tensor& scale_b, const SwizzleType swizzle_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          Tensor& out) {
#if defined(_WIN32) || (!defined(USE_ROCM) && !defined(USE_FBGEMM_GENAI))
  TORCH_CHECK_NOT_IMPLEMENTED(false, "MXFP4 scaling supported on ROCM and CUDA+FBGEMM_GENAI only");
#else
  // Restrictions:
  // A, B are FP4, scales are e8m0, A: shape K//32, B: K, N//32
  TORCH_CHECK_VALUE(mat_a.scalar_type() == at::kFloat4_e2m1fn_x2 && mat_b.scalar_type() == at::kFloat4_e2m1fn_x2, "mat_a and mat_b must be fp4 types, got: ",
      mat_a.scalar_type(), mat_b.scalar_type());

  // Packed FP4 format means actual-K = 2 * reported-K -- adjust
  auto K_multiplier = 2;
#ifdef USE_ROCM
  // AMD
  auto scale_a_elems = ceil_div<int64_t>(K_multiplier * mat_a.size(0), 32) * mat_a.size(1);
  auto scale_b_elems = ceil_div<int64_t>(K_multiplier * mat_b.size(1), 32) * mat_b.size(0);
#else
  // NVIDIA
  auto scale_a_elems = round_up<int64_t>(mat_a.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(K_multiplier * mat_a.size(1), 32), 4);
  auto scale_b_elems = round_up<int64_t>(mat_b.size(1), 128) * round_up<int64_t>(ceil_div<int64_t>(K_multiplier * mat_b.size(0), 32), 4);
#endif
  TORCH_CHECK_VALUE(scale_a_elems == scale_a.numel(),
         "For Blockwise scaling scale_a should have ", scale_a_elems, " elements, got: ", scale_a.numel());
  TORCH_CHECK_VALUE(scale_b_elems == scale_b.numel(),
         "For Blockwise scaling scale_b should have ", scale_b_elems, " elements, got: ", scale_b.numel());

#ifdef USE_ROCM
  // AMD
  TORCH_CHECK_VALUE(swizzle_a == SwizzleType::NO_SWIZZLE, "scale_a must not be swizzled (NO_SWIZZLE format)");
  TORCH_CHECK_VALUE(swizzle_b == SwizzleType::NO_SWIZZLE, "scale_b must not be swizzled (NO_SWIZZLE format)");
#else
  // NVIDIA
  TORCH_CHECK_VALUE(swizzle_a == SwizzleType::SWIZZLE_32_4_4, "scale_a must be swizzled to SWIZZLE_32_4_4 format");
  TORCH_CHECK_VALUE(swizzle_b == SwizzleType::SWIZZLE_32_4_4, "scale_b must be swizzled to SWIZZLE_32_4_4 format");
#endif

  TORCH_CHECK_VALUE(scale_a.is_contiguous() && scale_b.is_contiguous(),
        "For Blockwise scaling both scales should be contiguous");

  TORCH_CHECK_VALUE(out.scalar_type() == out_dtype, "expected out.scalar_type() to be ", out_dtype, ", but got ", out_dtype);

#ifdef USE_ROCM
  // AMD
  auto scaling_choice_a = ScalingType::BlockWise1x32;
  auto scaling_choice_b = ScalingType::BlockWise1x32;

#if ROCM_VERSION >= 70000
  TORCH_CHECK_NOT_IMPLEMENTED(at::detail::getCUDAHooks().isGPUArch({"gfx950"}),
              "Block-wise scaling for Float8_e8m0fnu is only supported on gfx950");

  TORCH_CHECK_VALUE(mat_a.size(0) % 32 == 0 && mat_a.size(1) % 32 == 0 &&
              mat_b.size(0) % 32 == 0 && mat_b.size(1) % 32 == 0,
              "Matrix dimensions must be multiples of 32 for block-wise scaling");

  TORCH_CHECK_VALUE(out.scalar_type() == ScalarType::BFloat16 ||
              out.scalar_type() == ScalarType::Half,
              "Block-wise scaling only supports BFloat16 or Half output types");
#endif

  return _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, false /* use_fast_accum */, out);
#else
  // NVIDIA
  // NOTE(slayton58): fbgemm_gpu::f4f4bf16 does *not* allow passing an output tensor,
  //                  but we have one we need to use. Two clear options are to copy into
  //                  our output (slow), or use a move-assignment-operator (faster).
  //                  However, the compiler can complain about the explicit move preventing
  //                  copy elision because the return from f4f4bf16 is a temporary object.
  //                  So we don't explicitly move, and trust the compiler here...
  //                  In the longer term this should be fixed on the FBGemm side.
  out = fbgemm_gpu::f4f4bf16(
      mat_a,
      mat_b.transpose(-2, -1),
      scale_a,
      scale_b,
      std::nullopt, /* global_scale */
      true          /* use_mx */
  );

  return out;
#endif
#endif
}

Tensor&
_scaled_nvfp4_nvfp4(
          const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a, const SwizzleType swizzle_a,
          const Tensor& scale_b, const SwizzleType swizzle_b,
          const std::optional<Tensor>& bias,
          const c10::ScalarType out_dtype,
          Tensor& out,
          const std::optional<Tensor>& global_scale_a = std::nullopt,
          const std::optional<Tensor>& global_scale_b = std::nullopt) {
#ifdef USE_ROCM
  TORCH_CHECK_NOT_IMPLEMENTED(false, "NVFP4 scaling not supported on ROCM");
#endif
  std::optional<Tensor> alpha = std::nullopt;
  // Note: "Or" here means that if only one scale is passed, we check for the other. Otherwise,
  //       if this is "And" we would silently do nothing in the case where one global scale is
  //       passed and not the other.
  if (global_scale_a.has_value() || global_scale_b.has_value()) {
    TORCH_CHECK_VALUE(global_scale_a.has_value(),
        "For two-level-scaled NVFP4, global_scale_a must have a value");
    TORCH_CHECK_VALUE(global_scale_b.has_value(),
        "For two-level-scaled NVFP4, global_scale_b must have a value");
    alpha = global_scale_a.value().mul(global_scale_b.value());
  }
  // Restrictions:
  // A, B are FP4, scales are e8m0, A: shape K//32, B: K, N//32
  // Scales must be swizzled
  TORCH_CHECK_VALUE(mat_a.scalar_type() == at::kFloat4_e2m1fn_x2 && mat_b.scalar_type() == at::kFloat4_e2m1fn_x2, "mat_a and mat_b must be fp4 types, got: ",
      mat_a.scalar_type(), mat_b.scalar_type());
  // Note: fp4x2 format, need to double the K dimension for checking purposes.
  auto scale_a_elems = round_up<int64_t>(mat_a.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_a.size(1) * 2, 16), 4);
  auto scale_b_elems = round_up<int64_t>(mat_b.size(1), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_b.size(0) * 2, 16), 4);
  TORCH_CHECK_VALUE(scale_a_elems == scale_a.numel(),
         "For Blockwise scaling scale_a should have ", scale_a_elems, " elements, got: ", scale_a.numel());
  TORCH_CHECK_VALUE(scale_b_elems == scale_b.numel(),
         "For Blockwise scaling scale_b should have ", scale_b_elems, " elements, got: ", scale_b.numel());

  TORCH_CHECK_VALUE(swizzle_a == SwizzleType::SWIZZLE_32_4_4, "scale_a must be swizzled to SWIZZLE_32_4_4 format");
  TORCH_CHECK_VALUE(swizzle_b == SwizzleType::SWIZZLE_32_4_4, "scale_b must be swizzled to SWIZZLE_32_4_4 format");

  TORCH_CHECK_VALUE(scale_a.is_contiguous() && scale_b.is_contiguous(),
        "For Blockwise scaling both scales should be contiguous");

  auto scaling_choice_a = ScalingType::BlockWise1x16;
  auto scaling_choice_b = ScalingType::BlockWise1x16;
  return _scaled_gemm(mat_a, mat_b, scale_a, scale_b, scaling_choice_a, scaling_choice_b, bias, false /* use_fast_accum */, out, alpha);
}


// V2: Computes matrix multiply + bias while applying scaling to input and output matrices
// Scales are only applicable when matrices are of Float8 type and assumed to be equal to 1.0 by default.
// If output matrix type is 16 or 32-bit type, scale_result is not applied.
// Known limitations:
//  - Only works if mat1 is row-major and mat2 is column-major
//  - Only works if matrices sizes are divisible by 32
//  - If 1-dimensional tensors are used then scale_a should be size = mat1.size(0)
//    and scale_b should have size = to mat2.size(1)
//  Arguments:
//    - `mat1`: the first operand of the matrix multiply, can be type `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `mat2`: the second operand of the matrix multiply, can be type `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `scale_a`: a tensor with the inverse scale of `mat1`, whose shape/strides/dtype depend on the scaling scheme
//    - `scale_recipe_a`: An integer corresponding to an enum describing the scaling scheme used for `scale_a`
//    - `swizzle_a`: An integer corresponding to a `SwizzleType` enum describing the swizzling scheme for `scale_a`
//    - `scale_b`: a tensor with the inverse scale of `mat2`, whose shape/strides/dtype depend on the scaling scheme
//    - `scale_recipe_b`: An integer corresponding to an enum describing the scaling scheme used for `scale_b`
//    - `swizzle_b`: An integer corresponding to a `SwizzleType` enum describing the swizzling scheme for `scale_b`
//    - `bias`: the bias, can be type `torch.float16` or `torch.bfloat16`
//    - `out_dtype`: the output dtype, can either be a float8 or a higher precision floating point type
//    - `use_fast_accum`: if true, enables fast float8 accumulation. Backends may ignore this option if not applicable.
//    - `out`: a reference to the output tensor
Tensor&
_scaled_mm_cuda_v2_out(
          const Tensor& mat_a, const Tensor& mat_b,
          ArrayRef<Tensor> scale_a,
          IntArrayRef scale_recipe_a,
          IntArrayRef swizzle_a,
          ArrayRef<Tensor> scale_b,
          IntArrayRef scale_recipe_b,
          IntArrayRef swizzle_b,
          const std::optional<Tensor>& bias,
          const std::optional<c10::ScalarType> out_dtype,
          IntArrayRef contraction_dim,
          bool use_fast_accum,
          Tensor& out) {
  // Check sizes
  bool allowed_device = _scaled_mm_allowed_device();
  TORCH_CHECK_NOT_IMPLEMENTED(allowed_device,
      "torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+");
  TORCH_CHECK_VALUE(mat_a.dim() == 2, "mat_a must be a matrix");
  TORCH_CHECK_VALUE(mat_b.dim() == 2, "mat_b must be a matrix");

  // If any of M, K, N is 0 - return early (the tensorwise/rowwise float8 gemm kernels
  // do not support this case).
  if (mat_a.size(0) == 0 || mat_a.size(1) == 0 || mat_b.size(1) == 0) {
    // `out` was created with `at::empty`. In the case where we are multiplying
    // MxK by KxN and K is the zero dim, we need to initialize here to properly
    // return a tensor of zeros.
    at::native::resize_output(out, {mat_a.size(0), mat_b.size(1)});
    if (mat_a.size(1) == 0) {
      out.zero_();
    }

    return out;
  }

  // Check if the input matrix sizes can be multiplied
  // - if optional contraction dims are provided, use those
  //   -- mostly for < 1B formats (i.e. nvfp4x2) where cheap .t() is not available.
  if (contraction_dim.size() > 0) {
    TORCH_CHECK_VALUE(contraction_dim.size() == 2, "contraction_dim must have exactly 2 elements");
    auto mat_a_dim = contraction_dim[0];
    auto mat_b_dim = contraction_dim[1];
    TORCH_CHECK_VALUE(
        mat_a.size(mat_a_dim) == mat_b.size(mat_b_dim), "mat_a and mat_b shapes cannot be multiplied (",
        mat_a.size(0), "x", mat_a.size(1), " and ", mat_b.size(0), "x", mat_b.size(1), ") ",
        "with contraction dims mat_a: ", mat_a_dim, ", mat_b: ", mat_b_dim);
  } else {
    TORCH_CHECK_VALUE(
        mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied (",
        mat_a.size(0), "x", mat_a.size(1), " and ", mat_b.size(0), "x", mat_b.size(1), ")");
  }

  // Handle fp4 packed-K dimension
  int K_multiplier = (mat_a.scalar_type() == ScalarType::Float4_e2m1fn_x2) ? 2 : 1;

  TORCH_CHECK_VALUE(!bias || bias->numel() == mat_b.sizes()[1], "Bias must be size ", mat_b.sizes()[1],
       " but got ", bias->numel());
  TORCH_CHECK_VALUE(
      K_multiplier * mat_a.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat_a.sizes()[0],
      "x",
      K_multiplier * mat_a.sizes()[1],
      ").");
  TORCH_CHECK_VALUE(K_multiplier * mat_b.sizes()[0] % 16 == 0 && mat_b.sizes()[1] % 16 == 0, "mat2 shape (", mat_b.sizes()[0], "x",
       mat_b.sizes()[1], ") must be divisible by 16");

  // TODO(slayton): Existing checks, not sure if they should really be here.
  TORCH_CHECK_VALUE(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK_VALUE(isFloat8Type(mat_a.scalar_type()) || mat_a.scalar_type() == ScalarType::Float4_e2m1fn_x2,
      "Expected mat_a to be Float8 or Float4_x2 matrix got ", mat_a.scalar_type());
  TORCH_CHECK_VALUE(isFloat8Type(mat_b.scalar_type()) || mat_b.scalar_type() == ScalarType::Float4_e2m1fn_x2,
      "Expected mat_b to be Float8 or Float4_x2 matrix got ", mat_b.scalar_type());
#ifndef USE_ROCM
  // Type restrictions imposed by CuBLASLt as of CUDA-12.1
  TORCH_CHECK_VALUE(mat_a.scalar_type() != ScalarType::Float8_e5m2 || mat_b.scalar_type() != ScalarType::Float8_e5m2,
        "Multiplication of two Float8_e5m2 matrices is not supported");
#endif
  if (use_fast_accum) {
    TORCH_CHECK_VALUE(mat_a.scalar_type() != ScalarType::Float4_e2m1fn_x2 && mat_b.scalar_type() != ScalarType::Float4_e2m1fn_x2, "`use_fast_accum` is not supported when `mat_a` or `mat_b` tensors have the `Float4_e2m1fn_x2` dtype.");
  }
#ifdef USE_ROCM
  if (mat_a.scalar_type() == ScalarType::Float4_e2m1fn_x2 || mat_b.scalar_type() == ScalarType::Float4_e2m1fn_x2) {
    TORCH_CHECK_NOT_IMPLEMENTED(ROCM_VERSION >= 70000,
        "Float4_e2m1fn_x2 is only supported for ROCm 7.0 and above");
  }
  if (mat_a.scalar_type() == ScalarType::Float8_e5m2 || mat_b.scalar_type() == ScalarType::Float8_e5m2) {
    TORCH_CHECK_NOT_IMPLEMENTED(ROCM_VERSION >= 60500,
        "Float8_e5m2 is only supported for ROCm 6.5 and above");
  }
  if (mat_a.scalar_type() == ScalarType::Float8_e4m3fn || mat_b.scalar_type() == ScalarType::Float8_e4m3fn) {
    TORCH_CHECK_NOT_IMPLEMENTED(ROCM_VERSION >= 60500,
        "Float8_e4m3fn is only supported for ROCm 6.5 and above");
  }
#endif
  if (bias) {
    TORCH_CHECK_VALUE(out.scalar_type() != kFloat,
        "Bias is not supported when out_dtype is set to Float32");

    TORCH_CHECK_VALUE(bias->scalar_type() == ScalarType::BFloat16 ||
                bias->scalar_type() == ScalarType::Half,
        "Bias must be BFloat16 or Half, but got ", bias->scalar_type());

    TORCH_CHECK_VALUE((out.scalar_type() != kFloat &&
                 out.scalar_type() != ScalarType::BFloat16) ||
                bias->scalar_type() == ScalarType::BFloat16,
        "Bias must be BFloat16 to compute ", out.scalar_type(),
        " output, but got ", bias->scalar_type());

    TORCH_CHECK_VALUE(out.scalar_type() != ScalarType::Half ||
                bias->scalar_type() == ScalarType::Half,
        "Bias must be Float16 to compute ", out.scalar_type(),
        " output, but got ", bias->scalar_type());
  }
  {
    auto bias_ = bias.value_or(Tensor());

    // NOLINTNEXTLINE(*c-array*)
    TensorArg targs[]{{out, "out", 0}, {mat_a, "mat_a", 1}, {mat_b, "mat_b", 2},
                      {bias_, "bias", 3}, {scale_a[0], "scale_a", 4}, {scale_b[0], "scale_b", 5}};
    checkAllSameGPU(__func__, targs);
  }

  auto out_dtype_ = out_dtype.value_or(at::ScalarType::BFloat16);

  // Conversion of implicitly-defined enums to explicit
  auto scale_recipe_a_enum = convert_int_to_enum<ScalingType>(scale_recipe_a);
  auto swizzle_a_enum = convert_int_to_enum<SwizzleType>(swizzle_a);
  auto scale_recipe_b_enum = convert_int_to_enum<ScalingType>(scale_recipe_b);
  auto swizzle_b_enum = convert_int_to_enum<SwizzleType>(swizzle_b);

  // at this point we can start working out what we want to be doing
  // Try to do as few steps as possible.
  // NOTE: support is deliberately sparse, can explicitly enumerate all combinations allowed.
  // Do this via a list of defined (name, acceptance, concrete_impl) tuples.
  bool found_impl = false;
  ScaledGemmImplementation gemm_impl = ScaledGemmImplementation::NONE;

  for (const auto& fn_entry : scale_kernel_dispatch) {
    const auto [name, accept_fn, scaled_gemm_impl] = fn_entry;
    bool ok = accept_fn(mat_a.scalar_type(),
                        scale_recipe_a_enum,
                        scale_a,
                        mat_b.scalar_type(),
                        scale_recipe_b_enum,
                        scale_b);
    if (ok) {
      gemm_impl = scaled_gemm_impl;
      found_impl = true;
      break;
    }
  }
  TORCH_CHECK_VALUE(
    found_impl,
    "Invalid scaling configuration.\n"
    "- For TensorWise scaling, a and b should be float8, scales should be float and singletons.\n"
    "- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (", mat_a.size(0), ", 1) and scale_b should be (1, ", mat_b.size(1), "), and both should be contiguous.\n"
    "- For BlockWise 1x128 scaling, a and b should be float8, scales should be float, scale_a should be (", mat_a.size(0), ", ", ceil_div<int64_t>(mat_a.size(1), 128), ") and scale_b should be (", ceil_div<int64_t>(mat_b.size(0), 128), ", ", mat_b.size(1), "), and both should be outer-dim-major.\n"
    "- For BlockWise 128x128 scaling, a and b should be float8, scales should be float, scale_a should be (", ceil_div<int64_t>(mat_a.size(0), 128), ", ", ceil_div<int64_t>(mat_a.size(1), 128), ") and scale_b should be (", ceil_div<int64_t>(mat_b.size(0), 128), ", ", ceil_div<int64_t>(mat_b.size(1), 128), "), and both should be near-inner-dim-major (with 16-byte aligned strides).\n"
    "- For Blockwise 1x32 scaling, a and b should be float8, scales should be float8_e8m0fnu, scale_a should have ", round_up<int64_t>(mat_a.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_a.size(1), 32), 4), " elements and scale_b should have ", round_up<int64_t>(mat_b.size(1), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_b.size(0), 32), 4), " elements, and both should be contiguous.\n"
    "- For Blockwise 1x16 scaling, a and b should be float4 (packed 2x), scales should be float8_e4m3fn, scale_a should have ", round_up<int64_t>(mat_a.size(0), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_a.size(1) * 2, 16), 4), " elements and scale_b should have ", round_up<int64_t>(mat_b.size(1), 128) * round_up<int64_t>(ceil_div<int64_t>(mat_b.size(0) * 2, 16), 4), " elements, and both should be contiguous.\n"
    "Got mat_a.dtype()=", mat_a.scalar_type(), ", scale_a[0].dtype()=", scale_a[0].scalar_type(), ", scale_a[0].size()=", scale_a[0].sizes(), ", scale_a[0].stride()=", scale_a[0].strides(), ", ",
    "mat_b.dtype()=", mat_b.scalar_type(), ", scale_b[0].dtype()=", scale_b[0].scalar_type(), ", scale_b[0].size()=", scale_b[0].sizes(), " and scale_b[0].stride()=", scale_b[0].strides()
  );

  at::native::resize_output(out, {mat_a.size(0), mat_b.size(1)});

  auto bias_ = bias.value_or(Tensor());

  // dispatch to appropriate lower-level calls for error checking & execution
  if (gemm_impl == ScaledGemmImplementation::TENSORWISE_TENSORWISE) {
    return _scaled_tensorwise_tensorwise(mat_a, mat_b, scale_a[0], scale_b[0], bias, out_dtype_, use_fast_accum, out);
  } else if (gemm_impl == ScaledGemmImplementation::ROWWISE_ROWWISE) {
    return _scaled_rowwise_rowwise(mat_a, mat_b, scale_a[0], scale_b[0], bias, out_dtype_, use_fast_accum, out);
  } else if (gemm_impl == ScaledGemmImplementation::BLOCK_128x128_1x128) {
    return _scaled_block128x128_block1x128(mat_a, mat_b, scale_a[0], scale_b[0], bias, out_dtype_, use_fast_accum, out);
  } else if (gemm_impl == ScaledGemmImplementation::BLOCK_1x128_128x128) {
    return _scaled_block1x128_block128x128(mat_a, mat_b, scale_a[0], scale_b[0], bias, out_dtype_, use_fast_accum, out);
  } else if (gemm_impl == ScaledGemmImplementation::BLOCK_1x128_1x128) {
    return _scaled_block1x128_block1x128(mat_a, mat_b, scale_a[0], scale_b[0], bias, out_dtype_, use_fast_accum, out);
  } else if (gemm_impl == ScaledGemmImplementation::MXFP8_MXFP8) {
    return _scaled_mxfp8_mxfp8(mat_a, mat_b, scale_a[0], swizzle_a_enum[0], scale_b[0], swizzle_b_enum[0], bias, out_dtype_, out);
  } else if (gemm_impl == ScaledGemmImplementation::NVFP4_NVFP4) {
    return _scaled_nvfp4_nvfp4(mat_a, mat_b, scale_a[0], swizzle_a_enum[0], scale_b[0], swizzle_b_enum[0], bias, out_dtype_, out,
                               scale_a[1], scale_b[1]);
  } else if (gemm_impl == ScaledGemmImplementation::NVFP4_NVFP4_SINGLE_SCALE) {
    return _scaled_nvfp4_nvfp4(mat_a, mat_b, scale_a[0], swizzle_a_enum[0], scale_b[0], swizzle_b_enum[0], bias, out_dtype_, out);
  } else if (gemm_impl == ScaledGemmImplementation::MXFP4_MXFP4) {
    return _scaled_mxfp4_mxfp4(mat_a, mat_b, scale_a[0], swizzle_a_enum[0], scale_b[0], swizzle_b_enum[0], bias, out_dtype_, out);
  } else {
    TORCH_CHECK_VALUE(false, "Invalid state - found an implementation, but not really");
  }
}

Tensor
_scaled_mm_cuda_v2(
          const Tensor& mat_a, const Tensor& mat_b,
          ArrayRef<Tensor> scale_a,
          IntArrayRef scale_recipe_a,
          IntArrayRef swizzle_a,
          ArrayRef<Tensor> scale_b,
          IntArrayRef scale_recipe_b,
          IntArrayRef swizzle_b,
          const std::optional<Tensor>& bias,
          const std::optional<c10::ScalarType> out_dtype,
          IntArrayRef contraction_dim,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));

  return _scaled_mm_cuda_v2_out(
                      mat_a, mat_b,
                      scale_a, scale_recipe_a, swizzle_a,
                      scale_b, scale_recipe_b, swizzle_b,
                      bias,
                      out_dtype,
                      contraction_dim,
                      use_fast_accum,
                      out);
}

} // namespace at::native

#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <c10/util/typeid.h>
#include <cstdint>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/ScaledBlasUtils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/xpu/grouped_mm/grouped_mm_api.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/StringUtil.h>

#include <cctype>
#include <cstdlib>
#include <string>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_grouped_mm_native.h>
#include <ATen/ops/_scaled_grouped_mm_v2_native.h>
#include <ATen/ops/_scaled_mm_native.h>
#endif

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace scaled_blas = at::native::scaled;
using scaled_blas::convert_int_to_enum;
using scaled_blas::ScaledGemmImplementation;

namespace at::native {

namespace {

void _check_scales_fp8_rowwise(
    const Tensor& mat,
    const Tensor& scale,
    const int dim,
    const int arg_idx,
    const int scale_multiplier = 1) {
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

void _check_scales_blocked(
    const Tensor& mat,
    const Tensor& scale,
    const int dim,
    const int arg_idx) {
  // if {mx,nv}fp4, will need to modify K later
  bool is_fp4 = (mat.scalar_type() == kFloat4_e2m1fn_x2);
  int blocksize = 32;
  // check for nvfp4 vs. mxfp4 to fix blocksize
  if (is_fp4 && scale.scalar_type() == kFloat8_e4m3fn) {
    blocksize = 16;
  }

  // Checks scales for 2d or 3d target tensors (`mat`).
  if (mat.dim() == 2) {
    // For MXFP8, 2d tensors have variable size groups represented as
    // subtensors, so we can't check the scale sizes without doing a d2h sync to
    // get the group sizes here.
    TORCH_CHECK(
        scale.dim() == mat.dim(),
        "for block-scaled, scale must have same number of dimensions as parent tensor, but got mat.dim() = ",
        mat.dim(),
        " and scale.dim() = ",
        scale.dim(),
        " for arg ",
        arg_idx);

    // LHS mat shape (M, total_K) -> scale shape (M,
    // rounded_up_per_group(K/blocksize)) RHS mat shape (total_K, N) -> scale
    // shape (rounded_up_per_group(K/blocksize), N)
    //   * weight is transposed prior to the call, scale stays non-transposed.
    bool LHS = arg_idx == 0;
    int dim_to_check = LHS ? 0 : 1;

    TORCH_CHECK(
        scale.size(dim_to_check) >= mat.size(dim_to_check),
        "for block-scaled, arg ",
        arg_idx,
        " tensor shape (",
        mat.size(0),
        ", ",
        mat.size(1),
        ") ",
        "must have scale.shape[",
        dim_to_check,
        "] >= ",
        mat.size(dim_to_check),
        " but got scale.shape=(",
        scale.size(0),
        ", ",
        scale.size(1),
        ")");
  } else {
    // For MXFP8, 3d tensors have static group sizes (stack of 2d tensors),
    // so we can check the exact expected scale sizes here without a d2h sync.
    // TODO: this is for 3d tensor in 2d-3d case specifically.
    // We'll need to support 3d-3d and 3d-2d cases once mxfp8/nvfp4 grouped gemm
    // supports them.
    int64_t G = mat.size(0);
    int64_t K = mat.size(1);
    if (is_fp4) {
      // FP4 packs 2 values into a single 8b word - the "real" K is 2x the
      // reported K. Reverse that adjustment.
      const int fp4_elems_per_byte = 2;
      K *= fp4_elems_per_byte;
    }
    int64_t blocked_scaled_K = (K + blocksize - 1) / blocksize;
    int64_t N = mat.size(2);

    // OneDNN expects scales as 3d tensor, shape (G, ceil(K/block_size), N).
    TORCH_CHECK(
        scale.dim() == mat.dim(),
        "for block-scaled 2d-3d grouped GEMM, the 3d tensor of shape (G,K,N) must have a 3d scale of shape (G, ceil(K/block_size), N),",
        "but scale is ",
        scale.dim(),
        "D for arg ",
        arg_idx);
    TORCH_CHECK(
        scale.size(0) == G && scale.size(1) == blocked_scaled_K &&
            scale.size(2) == N,
        "for block-scaled grouped GEMM, the tensor shape (",
        G,
        ", ",
        K,
        ", ",
        N,
        ") must have scale shape (",
        G,
        ",",
        blocked_scaled_K,
        ",",
        N,
        ")",
        " for arg ",
        arg_idx,
        ", got: ",
        scale.size(0),
        ", ",
        scale.size(1),
        ", ",
        scale.size(2));
  }
}

template <class ArrayType>
void check_swizzle(ArrayType& swizzle_enums) {
  for (auto swizzle_enum : swizzle_enums) {
    TORCH_CHECK(
        swizzle_enum == SwizzleType::NO_SWIZZLE,
        "XPU grouped GEMM currently only supports SWIZZLE_NONE swizzle type, but got swizzle type ",
        swizzle_enum);
  }
}

} // namespace

namespace {

using scaled_blas::ScaleKernelDispatchEntry;

std::array<ScaleKernelDispatchEntry, 4> scale_grouped_kernel_dispatch = {{
    {"rowwise_rowwise",
     scaled_blas::check_rowwise_recipe,
     ScaledGemmImplementation::ROWWISE_ROWWISE},
    {"mxfp8_mxfp8",
     scaled_blas::check_mxfp8_recipe,
     ScaledGemmImplementation::MXFP8_MXFP8},
    {"mxfp4_mxfp4",
     scaled_blas::check_mxfp4_recipe,
     ScaledGemmImplementation::MXFP4_MXFP4},
    {"nvfp4_nvfp4",
     scaled_blas::check_nvfp4_recipe,
     ScaledGemmImplementation::NVFP4_NVFP4}}};

} // anonymous namespace

// V2: Grouped scaled matrix multiply. Shape inference + output allocation and
// recipe-independent input validation run in
// TORCH_META_FUNC(_scaled_grouped_mm_v2); this impl handles strides/dtype
// validation that needs device context, the per-recipe scale-shape checks, and
// kernel dispatch.
TORCH_IMPL_FUNC(_scaled_grouped_mm_xpu_v2_out)
(const Tensor& mat_a,
 const Tensor& mat_b,
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
  TORCH_CHECK_VALUE(
      !check_valid_strides_and_return_transposed(mat_a),
      "Expected mat1 to not be transposed");
  TORCH_CHECK_VALUE(
      check_valid_strides_and_return_transposed(mat_b),
      "Expected mat2 to be transposed");
  TORCH_CHECK_VALUE(
      (mat_a.dim() == 2 && mat_b.dim() == 3),
      "Only 2d x 3d with offsets is supported for XPU scaled_grouped_mm for now");

  // Materialize the scale lists so the existing acceptance helpers (which take
  // ArrayRef<Tensor>) work unchanged.
  std::vector<Tensor> scale_a(scale_a_list.begin(), scale_a_list.end());
  std::vector<Tensor> scale_b(scale_b_list.begin(), scale_b_list.end());
  ArrayRef<Tensor> scale_a_ref(scale_a);
  ArrayRef<Tensor> scale_b_ref(scale_b);

  // The output has already been sized by the structured-op meta function.
  Tensor& out_mut = const_cast<Tensor&>(out);

  // Conversion of implicitly-defined enums to explicit
  auto swizzle_a_enum = convert_int_to_enum<SwizzleType>(swizzle_a);
  auto swizzle_b_enum = convert_int_to_enum<SwizzleType>(swizzle_b);

  // swizze checks
  check_swizzle(swizzle_a_enum);
  check_swizzle(swizzle_b_enum);

  // at this point we can start working out what we want to be doing
  // Try to do as few steps as possible.
  // NOTE: support is deliberately sparse, can explicitly enumerate all
  // combinations allowed. Do this via a list of defined (name, acceptance,
  // concrete_impl) tuples.
  auto scale_recipe_a_enum = convert_int_to_enum<ScalingType>(scale_recipe_a);
  auto scale_recipe_b_enum = convert_int_to_enum<ScalingType>(scale_recipe_b);
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
  TORCH_CHECK_VALUE(
      gemm_impl != ScaledGemmImplementation::NONE,
      "No gemm implementation was found");

  std::optional<Tensor> alpha = std::nullopt;
  switch (gemm_impl) {
    case ScaledGemmImplementation::ROWWISE_ROWWISE: {
      const int scale_multiplier =
          (mat_a.dim() == 2 && mat_b.dim() == 2) ? offs->size(0) : 1;
      _check_scales_fp8_rowwise(
          mat_a, scale_a[0], 0 /* dim */, 0 /* arg_idx */, scale_multiplier);
      _check_scales_fp8_rowwise(
          mat_b, scale_b[0], 1 /* dim */, 1 /* arg_idx */, scale_multiplier);
      break;
    }
    case ScaledGemmImplementation::MXFP8_MXFP8:
    case ScaledGemmImplementation::MXFP4_MXFP4:
    case ScaledGemmImplementation::NVFP4_NVFP4: {
      // scale shape checks
      _check_scales_blocked(mat_a, scale_a[0], 0 /* dim */, 0 /* arg_idx */);
      _check_scales_blocked(mat_b, scale_b[0], 1 /* dim */, 1 /* arg_idx */);
      if (gemm_impl == ScaledGemmImplementation::NVFP4_NVFP4) {
        alpha = scale_a[1].mul(scale_b[1]);
      }
      break;
    }
    default:
      TORCH_CHECK_NOT_IMPLEMENTED(
          false,
          "_scaled_grouped_mm_xpu_v2 is in an inconsistent state - should never reach here");
  }
  at::native::onednn::scaled_grouped_matmul(
      mat_a,
      mat_b,
      scale_a[0],
      scale_b[0],
      scale_recipe_a_enum[0],
      scale_recipe_b_enum[0],
      *offs,
      out_mut,
      alpha);
  return;
}

namespace {

// TORCH_XPU_GROUPED_MM_BACKEND = auto (default) | sycltla | onednn | fallback,
// read on every call.
//   auto      SYCL-TLA if it can serve the call, else oneDNN, else the
//             per-group fallback (the order of the checks in _grouped_mm_xpu)
//   sycltla   force SYCL-TLA, onednn force oneDNN: the per-group fallback
//             (with a one-time warning) for calls the backend cannot serve
//   fallback  always the per-group fallback
// Like _grouped_mm_cuda, the checks use only dtype, layout and alignment,
// never the routed row counts, so every rank of an expert-parallel job runs
// the same kernel for the same op.
enum class GroupedMMBackend { Auto, SyclTla, OneDnn, Fallback };

GroupedMMBackend requested_grouped_mm_backend() {
  const char* v = std::getenv("TORCH_XPU_GROUPED_MM_BACKEND");
  const std::string s = (v == nullptr) ? "" : v;
  if (s.empty() || s == "auto") {
    return GroupedMMBackend::Auto;
  }
  if (s == "sycltla" || s == "sycl-tla") {
    return GroupedMMBackend::SyclTla;
  }
  if (s == "onednn") {
    return GroupedMMBackend::OneDnn;
  }
  if (s == "fallback") {
    return GroupedMMBackend::Fallback;
  }
  TORCH_WARN_ONCE(
      "Ignoring unknown TORCH_XPU_GROUPED_MM_BACKEND='",
      s,
      "'; expected auto, sycltla, onednn or fallback.");
  return GroupedMMBackend::Auto;
}

// SYCL-TLA grouped GEMM (torch-xpu-ops): "" if it can serve the call, else
// the reason.
std::string sycltla_why_not(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    c10::ScalarType out_dtype,
    const Tensor& out) {
  if (!offs.has_value()) {
    return "no offs (3D x 3D batched)";
  }
  if (bias.has_value()) {
    return "bias";
  }
  if (out_dtype != mat_a.scalar_type()) {
    return "out_dtype differs from input dtype";
  }
  std::string why;
  sycltla::can_use_grouped_mm(mat_a, mat_b, *offs, out, &why);
  return why;
}

// oneDNN grouped matmul (pytorch/pytorch#195427, #195428): 2D x 3D and
// 2D x 2D, same dtype for inputs and output, no bias.
std::string onednn_why_not(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {
  const bool a_b_and_out_are_same_type = (mat_a.dtype() == mat_b.dtype()) &&
      (!out_dtype.has_value() || mat_a.dtype() == *out_dtype);
  if (!a_b_and_out_are_same_type) {
    return "dtypes differ";
  }
  if (mat_a.dim() != 2) {
    return "mat_a is not 2D";
  }
  if (bias.has_value()) {
    return "bias";
  }
  return "";
}

// Log de-duplication key for a dispatch decision: the reason without its
// numbers, plus the dims that do not change with routing.
std::string grouped_mm_log_key(
    const char* path,
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::string& reason) {
  std::string key = c10::str(path, "|");
  for (char ch : reason) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) {
      key += ch;
    }
  }
  return c10::str(
      key,
      "|",
      mat_a.dim(),
      "x",
      mat_b.dim(),
      "|",
      mat_a.scalar_type(),
      "|",
      (mat_a.dim() == 2 && mat_b.dim() == 2) ? mat_a.size(0) : mat_a.size(-1),
      "|",
      mat_b.size(-1));
}

} // namespace

// torch._grouped_mm on XPU, structured like _grouped_mm_cuda: validate,
// allocate the output, then a grouped GEMM kernel (SYCL-TLA from
// torch-xpu-ops, else oneDNN) or the generic per-group fallback. The kernels
// serve the shapes autograd produces for a MoE expert layer:
//   forward      mat_a [M, K] x mat_b [E, K, N]              -> [M, N]
//   grad_input   mat_a [M, N] x mat_b.transpose(-2, -1)      -> [M, K]
//   grad_weight  mat_a.t() [K, M] x grad [M, N]              -> [E, K, N]
//
// TORCH_XPU_GROUPED_MM_LOG=1 (each distinct decision once) or =2 (every call)
// prints which kernel each call takes, and why, to stderr.
Tensor _grouped_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto out_dtype_ =
      _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out =
      create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  const GroupedMMBackend req = requested_grouped_mm_backend();
  std::string reason;
  if (req == GroupedMMBackend::Fallback) {
    reason = "TORCH_XPU_GROUPED_MM_BACKEND=fallback";
  } else {
    // fast paths, no d2h sync needed
    if (req == GroupedMMBackend::Auto || req == GroupedMMBackend::SyclTla) {
      const std::string why =
          sycltla_why_not(mat_a, mat_b, offs, bias, out_dtype_, out);
      if (why.empty()) {
        sycltla::grouped_mm(mat_a, mat_b, *offs, out); // logs its own plan
        return out;
      }
      reason = "sycltla: " + why;
    }
    if (req == GroupedMMBackend::Auto || req == GroupedMMBackend::OneDnn) {
      const std::string why = onednn_why_not(mat_a, mat_b, bias, out_dtype);
      if (why.empty()) {
        if (sycltla::grouped_mm_log_level() > 0) {
          sycltla::log_grouped_mm(
              grouped_mm_log_key("onednn", mat_a, mat_b, reason),
              c10::str(
                  "oneDNN grouped matmul",
                  reason.empty() ? "" : " (",
                  reason,
                  reason.empty() ? "" : ")",
                  "  a",
                  mat_a.sizes(),
                  " b",
                  mat_b.sizes(),
                  " -> out",
                  out.sizes()));
        }
        at::native::onednn::scaled_grouped_matmul(
            mat_a,
            mat_b,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            offs.value(),
            out);
        return out;
      }
      reason += (reason.empty() ? "" : "; ") + ("onednn: " + why);
    }
    if (req != GroupedMMBackend::Auto) {
      TORCH_WARN_ONCE(
          "TORCH_XPU_GROUPED_MM_BACKEND: the forced grouped GEMM cannot serve "
          "some calls (",
          reason,
          "); using the per-group fallback for those.");
    }
  }
  if (sycltla::grouped_mm_log_level() > 0) {
    sycltla::log_grouped_mm(
        grouped_mm_log_key("fallback", mat_a, mat_b, reason),
        c10::str(
            "fallback per-group mm: ",
            reason,
            "  a",
            mat_a.sizes(),
            " b",
            mat_b.sizes(),
            " -> out",
            out.sizes()));
  }
  _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  return out;
}

} // namespace at::native

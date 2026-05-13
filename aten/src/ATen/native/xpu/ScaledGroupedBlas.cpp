#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/xpu/ScaledGroupedMM.h>

namespace at::native {

namespace {

// Scale validation for FP8 rowwise (float32) scales.
// Mirrors _check_scales_fp8_rowwise from cuda/GroupedBlas.cpp:301-336.
void _check_scales_fp8_rowwise(
    const Tensor& mat,
    const Tensor& scale,
    const int dim,
    const int arg_idx,
    const int scale_multiplier = 1) {
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

} // anonymous namespace

Tensor _scaled_grouped_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum) {
  // --- Validation (mirroring CUDA GroupedBlas.cpp:428-478) ---

  // Stride checks
  TORCH_CHECK_VALUE(
      !check_valid_strides_and_return_transposed(mat_a),
      "Expected mat1 to not be transposed");
  TORCH_CHECK_VALUE(
      check_valid_strides_and_return_transposed(mat_b),
      "Expected mat2 to be transposed");

  // Dimension checks
  TORCH_CHECK_VALUE(
      mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK_VALUE(
      mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");

  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;

  // Contraction dimension match
  if (!a_is_2d || !b_is_2d) {
    TORCH_CHECK_VALUE(
        mat_a.size(-1) == mat_b.size(-2),
        "contraction dimension of mat_a and mat_b must match");
  }

  // Divisibility by 16
  TORCH_CHECK_VALUE(
      mat_a.size(-1) % 16 == 0,
      "Expected trailing dimension of mat_a to be divisible by 16 "
      "but got mat1 shape: (",
      mat_a.sizes(),
      ").");
  TORCH_CHECK_VALUE(
      mat_b.size(-2) % 16 == 0 && mat_b.size(-1) % 16 == 0,
      "Expected mat_b shape to be divisible by 16 "
      "but got mat_b shape: (",
      mat_b.sizes(),
      ").");

  // Unsupported features
  TORCH_CHECK_VALUE(!bias.has_value(), "Bias not supported yet");
  TORCH_CHECK_VALUE(
      !scale_result.has_value(), "Scale result not supported yet");

  // Offsets
  TORCH_CHECK_VALUE(
      offs.has_value() == (a_is_2d || b_is_2d),
      "Have to provide offsets if there is a 2d matrix");
  if (offs.has_value()) {
    TORCH_CHECK_VALUE(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK_VALUE(
        offs->dtype() == at::kInt, "Offsets have to be int32");
  }

  // Scales: only float32 rowwise supported on XPU
  TORCH_CHECK_VALUE(
      scale_a.scalar_type() == at::kFloat &&
          scale_b.scalar_type() == at::kFloat,
      "XPU _scaled_grouped_mm only supports float32 rowwise scales. "
      "Got scale_a: ",
      scale_a.scalar_type(),
      ", scale_b: ",
      scale_b.scalar_type());

  // Scale shape validation
  const int scale_multiplier =
      (a_is_2d && b_is_2d) ? offs->size(0) : 1;
  _check_scales_fp8_rowwise(mat_a, scale_a, 0, 0, scale_multiplier);
  _check_scales_fp8_rowwise(mat_b, scale_b, 1, 1, scale_multiplier);

  // Output dtype
  const auto out_dtype_ = out_dtype.value_or(at::kBFloat16);
  TORCH_CHECK_VALUE(
      out_dtype_ == at::kBFloat16,
      "Only bf16 high precision output types are supported for "
      "scaled grouped gemm");

  // Create output tensor
  Tensor out =
      create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  // --- Dispatch ---
  TORCH_CHECK(
      at::native::xpu::is_scaled_grouped_mm_available(),
      "_scaled_grouped_mm: XPU kernel not available (requires SYCLTLA)");

  at::native::xpu::f8f8bf16_scaled_grouped_mm(
      mat_a, mat_b, scale_a, scale_b, offs, out);

  return out;
}

} // namespace at::native

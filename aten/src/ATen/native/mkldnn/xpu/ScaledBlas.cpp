#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/BlasBackend.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/ceil_div.h>
#include <ATen/native/Resize.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/xpu/Blas.h>
#include <torch/library.h>

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

namespace at::native {

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace {
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
 * - Else if scale.dim() == 2 && scale.size(0) == outer_dim && scale.size(1) ==
 * 1:
 *   - Returns RowWise.
 *
 * - Otherwise:
 *   - Returns Error.
 */

bool is_tensorwise_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return at::isFloat8Type(t.scalar_type()) &&
      scale.scalar_type() == at::kFloat && scale.numel() == 1;
}

bool is_rowwise_scaling(const at::Tensor& t, const at::Tensor& scale) {
  return (
      at::isFloat8Type(t.scalar_type()) && scale.scalar_type() == at::kFloat &&
      scale.dim() == 2 && scale.size(0) == t.size(0) && scale.size(1) == 1 &&
      scale.is_contiguous());
}

bool is_desired_scaling(
    const at::Tensor& t,
    const at::Tensor& scale,
    ScalingType desired_scaling) {
  auto result = desired_scaling == ScalingType::TensorWise
      ? is_tensorwise_scaling(t, scale)
      : is_rowwise_scaling(t, scale);
  return result;
}

std::pair<ScalingType, ScalingType> get_joint_scaling(
    std::initializer_list<std::pair<ScalingType, ScalingType>> options,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b) {
  for (auto [lhs, rhs] : options) {
    if (is_desired_scaling(a, scale_a, lhs) &&
        is_desired_scaling(b.t(), scale_b.t(), rhs)) {
      return {lhs, rhs};
    }
  }
  TORCH_CHECK(
      false,
      "Invalid scaling configuration.\n"
      "- For TensorWise scaling, a and b should be float8, scales should be float and singletons.\n"
      "- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (",
      a.size(0),
      ", 1) and scale_b should be (1, ",
      b.size(1),
      "), and both should be contiguous.\n"
      "Got a.dtype()=",
      a.scalar_type(),
      ", scale_a.dtype()=",
      scale_a.scalar_type(),
      ", scale_a.size()=",
      scale_a.sizes(),
      ", scale_a.stride()=",
      scale_a.strides(),
      ", ",
      "b.dtype()=",
      b.scalar_type(),
      ", scale_b.dtype()=",
      scale_b.scalar_type(),
      ", scale_b.size()=",
      scale_b.sizes(),
      " and scale_b.stride()=",
      scale_b.strides());
}

Tensor& _scaled_gemm(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const ScalingType scaling_choice_a,
    const ScalingType scaling_choice_b,
    const std::optional<Tensor>& bias,
    const bool use_fast_accum,
    Tensor& out,
    const std::optional<Tensor>& alpha = std::nullopt) {
  // TODO: scale_result and alpha is not defined or used!
  std::optional<Tensor> scaled_result = std::nullopt;
  at::native::onednn::scaled_matmul(
      mat1,
      mat2,
      out,
      scale_a,
      scale_b,
      scaling_choice_a,
      scaling_choice_b,
      bias,
      scaled_result,
      use_fast_accum);

  return out;
}

} // namespace

// Computes matrix multiply + bias while applying scaling to input and output
// matrices Scales are only applicable when matrices are of Float8 type and
// assumed to be equal to 1.0 by default. If output matrix type is 16 or 32-bit
// type, scale_result is not applied. Known limitations:
//  - Only works if mat1 is row-major and mat2 is column-major
//  - Only works if matrices sizes are divisible by 32
//  - If 1-dimensional tensors are used then scale_a should be size =
//  mat1.size(0)
//    and scale_b should have size = to mat2.size(1)
//  Arguments:
//    - `mat1`: the first operand of the matrix multiply, can be type
//    `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `mat2`: the second operand of the matrix multiply, can be type
//    `torch.float8_e4m3fn` or `torch.float8_e5m2`
//    - `bias`: the bias, can be type `torch.float16` or `torch.bfloat16`
//    - `out_dtype`: the output dtype, can either be a float8 or a higher
//    precision floating point type
//    - `scale_a`: a tensor with the inverse scale of `mat1`, whose
//    shape/strides/dtype depend on the scaling scheme
//    - `scale_b`: a tensor with the inverse scale of `mat2`, whose
//    shape/strides/dtype depend on the scaling scheme
//    - `scale_result`: a scalar tensor with the scale of the output, only
//    utilized if the output is a float8 type
//    - `use_fast_accum`: Not applicable for XPU. For now, it should always be
//    false.
//    - `out`: a reference to the output tensor

Tensor& _scaled_mm_out_xpu(
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum,
    Tensor& out) {
  // Note: fast_accum is not supported in XPU for now.
  TORCH_CHECK(!use_fast_accum, "fast_accum is not supported in XPU for now.");

  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");

  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");

  // Check what type of scaling we are doing based on inputs. This list is
  // sorted by decreasing priority.

  // List of supported datatypes for XPU with oneDNN:
  // https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html#data-types
  auto [scaling_choice_a, scaling_choice_b] = get_joint_scaling(
      {
          std::make_pair(ScalingType::TensorWise, ScalingType::TensorWise),
          std::make_pair(ScalingType::RowWise, ScalingType::RowWise),
      },
      mat1,
      mat2,
      scale_a,
      scale_b);
  TORCH_CHECK(
      !scale_result ||
          (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
      "scale_result must be a float scalar");
  TORCH_CHECK(
      !bias || bias->numel() == mat2.sizes()[1],
      "Bias must be size ",
      mat2.sizes()[1],
      " but got ",
      bias->numel());
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ").");
  TORCH_CHECK(
      mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0,
      "mat2 shape (",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ") must be divisible by 16");
  // Check types
  TORCH_CHECK(
      !out_dtype || *out_dtype == out.scalar_type(),
      "out_dtype must match output matrix type");
  TORCH_CHECK(
      at::isFloat8Type(mat1.scalar_type()),
      "Expected mat1 to be Float8 matrix got ",
      mat1.scalar_type());
  TORCH_CHECK(
      at::isFloat8Type(mat2.scalar_type()),
      "Expected mat2 to be Float8 matrix got ",
      mat2.scalar_type());
  // TODO: oneDNN Currently only supports e4m3 with group scales on BMG. Not
  // support 2D scales, only 1D. Needs to add more checks there.

  if (bias) {
    TORCH_CHECK(
        bias->scalar_type() == kFloat ||
            bias->scalar_type() == c10::ScalarType::BFloat16 ||
            bias->scalar_type() == c10::ScalarType::Half,
        "Bias must be Float32 or BFloat16 or Half, but got ",
        bias->scalar_type());
  }

  {
    auto bias_ = bias.value_or(Tensor());
    auto scale_result_ = scale_result.value_or(Tensor());

    // NOLINTNEXTLINE(*c-array*)
    TensorArg targs[]{
        {out, "out", 0},
        {mat1, "mat1", 1},
        {mat2, "mat2", 2},
        {bias_, "bias", 3},
        {scale_a, "scale_a", 4},
        {scale_b, "scale_b", 5},
        {scale_result_, "scale_result", 6}};
    checkAllSameGPU(__func__, targs);
  }

  // Validation checks have passed lets resize the output to actual size
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  // If any of M, K, N is 0 - return early (the tensorwise/rowwise float8 gemm
  // kernels do not support this case).
  if (mat1_sizes[0] == 0 || mat1_sizes[1] == 0 || mat2_sizes[1] == 0) {
    // `out` was created with `at::empty`. In the case where we are multiplying
    // MxK by KxN and K is the zero dim, we need to initialize here to properly
    // return a tensor of zeros.
    if (mat1_sizes[1] == 0) {
      out.zero_();
    }

    return out;
  }

  // TODO: Scale_result is not supported by now!!
  return _scaled_gemm(
      mat1,
      mat2,
      scale_a,
      scale_b,
      scaling_choice_a,
      scaling_choice_b,
      bias,
      use_fast_accum,
      out);
}

Tensor _scaled_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return _scaled_mm_out_xpu(
      mat_a,
      mat_b,
      scale_a,
      scale_b,
      bias,
      scale_result,
      out_dtype,
      use_fast_accum,
      out);
}

} // namespace at::native

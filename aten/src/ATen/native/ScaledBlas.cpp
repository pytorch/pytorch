#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Config.h>

#include <ATen/native/mkldnn/Matmul.h>
#include <ATen/native/mkldnn/Linear.h>
#include <ATen/native/Resize.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/BlasBackend.h>
#include <ATen/native/ScaledBlasUtils.h>
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/addmv.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/dot.h>
#include <ATen/ops/dot_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul_cpu_dispatch.h>
#include <ATen/ops/mv_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/vdot_native.h>
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/_scaled_mm_v2_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/matmul.h>
#endif

namespace at::native {

using at::blas::ScalingType;
using at::blas::SwizzleType;

namespace {

void invalid_scaling_config(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b) {
  TORCH_CHECK_VALUE(
    false,
    "Invalid scaling configuration.\n"
    "- For TensorWise scaling, a and b should be float8, scales should be float and singletons.\n"
    "- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (",
    mat_a.size(0),
    ", 1) and scale_b should be (1, ",
    mat_b.size(1),
    "), and both should be contiguous.\n"
    "Got mat_a.dtype()=",
    mat_a.scalar_type(),
    ", scale_a[0].dtype()=",
    scale_a[0].scalar_type(),
    ", scale_a[0].size()=",
    scale_a[0].sizes(),
    ", scale_a[0].stride()=",
    scale_a[0].strides(),
    ", ",
    "mat_b.dtype()=",
    mat_b.scalar_type(),
    ", scale_b[0].dtype()=",
    scale_b[0].scalar_type(),
    ", scale_b[0].size()=",
    scale_b[0].sizes(),
    " and scale_b[0].stride()=",
    scale_b[0].strides());
}

/*
 * Scaling Type Determination:
 * ---------------------------
 * Conditions and corresponding Scaling Types:
 *
 * - If scale.numel() == 1:
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
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b) {
  for (auto [lhs, rhs] : options) {
    if (is_desired_scaling(mat_a, scale_a, lhs) &&
        is_desired_scaling(mat_b.t(), scale_b.t(), rhs)) {
      return {lhs, rhs};
    }
  }
  invalid_scaling_config(mat_a, mat_b, scale_a, scale_b);
  return {};
}

} // namespace

static Tensor&
_scaled_mm_out_cpu_emulated(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          const at::scaled::ScaledGemmImplementation gemm_impl,
          Tensor& out) {
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a matrix");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a matrix");
  TORCH_CHECK(
      mat_a.sizes()[1] == mat_b.sizes()[0], "mat_a and mat_b shapes cannot be multiplied (",
      mat_a.sizes()[0], "x", mat_a.sizes()[1], " and ", mat_b.sizes()[0], "x", mat_b.sizes()[1], ")");

  TORCH_CHECK_VALUE(gemm_impl == at::scaled::ScaledGemmImplementation::TENSORWISE_TENSORWISE ||
    gemm_impl == at::scaled::ScaledGemmImplementation::ROWWISE_ROWWISE, "Unsupported scaling implementation");

  if (gemm_impl == at::scaled::ScaledGemmImplementation::TENSORWISE_TENSORWISE) {
    // Restrictions:
    // A, B are FP8, scales are fp32
    TORCH_CHECK_VALUE(
    isFloat8Type(mat_a.scalar_type()) && isFloat8Type(mat_b.scalar_type()),
    "mat_a and mat_b must be fp8 types, got: ",
    mat_a.scalar_type(),
    mat_b.scalar_type());
    TORCH_CHECK_VALUE(
        scale_a.numel() == 1 && scale_a.scalar_type() == kFloat,
        "scale_a must have 1 Float element");
    TORCH_CHECK_VALUE(
        scale_b.numel() == 1 && scale_b.scalar_type() == kFloat,
        "scale_b must have 1 Float element");
  }
  else {
    // Restrictions:
    // A, B are FP8, scales are fp32, shape M/N for A/B
    TORCH_CHECK_VALUE(
        isFloat8Type(mat_a.scalar_type()) && isFloat8Type(mat_b.scalar_type()),
        "mat_a and mat_b must be fp8 types, got: ",
        mat_a.scalar_type(),
        mat_b.scalar_type());
    TORCH_CHECK_VALUE(
        scale_a.size(0) == mat_a.size(0) && scale_a.size(1) == 1,
        "scale_a must have shape [",
        mat_a.size(0),
        ", 1], got [",
        scale_a.sizes(),
        "]");
    TORCH_CHECK_VALUE(
        scale_a.numel() == mat_a.size(0) && scale_a.scalar_type() == kFloat,
        "scale_a must have ",
        mat_a.size(0),
        " Float elements, got ",
        scale_a.numel())
    TORCH_CHECK_VALUE(
        scale_b.numel() == mat_b.size(1) && scale_b.scalar_type() == kFloat,
        "scale_b must have ",
        mat_b.size(1),
        " Float elements, got ",
        scale_b.numel())

    TORCH_CHECK_VALUE(
        scale_a.stride(1) == 1,
        "expected scale_a.stride(1) to be 1, but got ",
        scale_a.stride(1));
    TORCH_CHECK_VALUE(
        scale_b.stride(1) == 1,
        "expected scale_b.stride(1) to be 1, but got ",
        scale_b.stride(1));
  }

  TORCH_CHECK(
      !scale_result ||
          (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
      "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat_b.sizes()[1], "Bias must be size ", mat_b.sizes()[1],
       " but got ", bias->numel());

  // Check types
  TORCH_CHECK(isFloat8Type(mat_a.scalar_type()), "Expected mat_a to be Float8 matrix got ", mat_a.scalar_type());
  TORCH_CHECK(isFloat8Type(mat_b.scalar_type()), "Expected mat_b to be Float8 matrix got ", mat_b.scalar_type());

  auto mat_a_cont = mat_a.contiguous();
  auto mat_b_cont = mat_b.contiguous();
  IntArrayRef mat_a_sizes = mat_a_cont.sizes();
  IntArrayRef mat_b_sizes = mat_b_cont.sizes();
  at::native::resize_output(out, {mat_a_sizes[0], mat_b_sizes[1]});

  float output_scale = 1.0f;
  if (scale_result.has_value() &&
      (*out_dtype == ScalarType::Float8_e4m3fn ||
       *out_dtype == ScalarType::Float8_e5m2)) {
    output_scale = scale_result.value().item<float>();
  }

  at::Tensor fp32_mat_a;
  at::Tensor fp32_mat_b;
  if (gemm_impl == at::scaled::ScaledGemmImplementation::TENSORWISE_TENSORWISE) {
    fp32_mat_a = at::mul(mat_a_cont.to(kFloat), scale_a.item<float>());
    fp32_mat_b = at::mul(mat_b_cont.to(kFloat), scale_b.item<float>());
  }
  else {
    fp32_mat_a = at::mul(mat_a_cont.to(kFloat), scale_a);
    fp32_mat_b = at::mul(mat_b_cont.to(kFloat), scale_b);
  }

  auto out_tmp = at::matmul(fp32_mat_a, fp32_mat_b);
  if (bias) {
    out_tmp.add_(bias.value());
  }
  if (*out_dtype == ScalarType::Float8_e4m3fn ||
      *out_dtype == ScalarType::Float8_e5m2) {
    out_tmp = at::mul(out_tmp, 1 / output_scale);
  }
  out_tmp = out_tmp.to(out.scalar_type());
  out.copy_(out_tmp);
  return out;
}

Tensor&
_scaled_mm_out_cpu(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
#if AT_MKLDNN_ENABLED() && !defined(__powerpc__)
  if (at::globalContext().userEnabledMkldnn()) {
    bool mixed_dtype = mat_a.scalar_type() != mat_b.scalar_type();
    if ((!mixed_dtype && cpuinfo_has_x86_amx_int8()) ||
        (mixed_dtype && cpuinfo_has_x86_amx_fp16())) {
      return mkldnn_scaled_mm(
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
  }
#endif
  {
    TORCH_CHECK_VALUE(
      !out_dtype || *out_dtype == out.scalar_type(),
      "out_dtype must match output matrix type");

    const auto [scaling_choice_a, scaling_choice_b] = get_joint_scaling(
        {
            std::make_pair(ScalingType::TensorWise, ScalingType::TensorWise),
            std::make_pair(ScalingType::RowWise, ScalingType::RowWise),
        },
        mat_a,
        mat_b,
        scale_a,
        scale_b);

    at::scaled::ScaledGemmImplementation gemm_impl{at::scaled::ScaledGemmImplementation::NONE};

    if (scaling_choice_a == ScalingType::TensorWise && scaling_choice_b == ScalingType::TensorWise) {
      gemm_impl = at::scaled::ScaledGemmImplementation::TENSORWISE_TENSORWISE;
    }
    else if (scaling_choice_a == ScalingType::RowWise && scaling_choice_b == ScalingType::RowWise) {
      gemm_impl = at::scaled::ScaledGemmImplementation::ROWWISE_ROWWISE;
    }

  return _scaled_mm_out_cpu_emulated(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, gemm_impl, out);
  }
}

Tensor
_scaled_mm_cpu(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return _scaled_mm_out_cpu(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

using acceptance_fn = std::function<bool(
    c10::ScalarType,
    std::vector<ScalingType>&,
    ArrayRef<Tensor>&,
    c10::ScalarType,
    std::vector<ScalingType>&,
    ArrayRef<Tensor>&)>;

using at::scaled::convert_int_to_enum;
using at::scaled::ScaledGemmImplementation;

std::array<std::tuple<std::string, acceptance_fn, at::scaled::ScaledGemmImplementation>, 2>
    scale_kernel_dispatch = {{
      {"tensorwise_tensorwise",
       at::scaled::check_tensorwise_recipe,
       ScaledGemmImplementation::TENSORWISE_TENSORWISE},
      {"rowwise_rowwise",
       at::scaled::check_rowwise_recipe,
       ScaledGemmImplementation::ROWWISE_ROWWISE},

  }};

Tensor& _scaled_mm_cpu_v2_out(
    const Tensor& mat_a,
    const Tensor& mat_b,
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

  TORCH_CHECK_VALUE(
    !bias || bias->numel() == mat_b.sizes()[1],
    "Bias must be size ",
    mat_b.sizes()[1],
    " but got ",
    bias->numel());

  TORCH_CHECK_VALUE(
    !out_dtype || *out_dtype == out.scalar_type(),
    "out_dtype must match output matrix type");

  if (bias) {
    TORCH_CHECK_VALUE(
        bias->scalar_type() == kFloat ||
            bias->scalar_type() == c10::ScalarType::BFloat16 ||
            bias->scalar_type() == c10::ScalarType::Half,
        "Bias must be Float32 or BFloat16 or Half, but got ",
        bias->scalar_type());
  }

  // Align with CUDA's default out to be bf16
  const auto out_dtype_ = out_dtype.value_or(c10::ScalarType::BFloat16);

  // Conversion of implicitly-defined enums to explicit
  auto scale_recipe_a_enum = convert_int_to_enum<ScalingType>(scale_recipe_a);
  auto swizzle_a_enum = convert_int_to_enum<SwizzleType>(swizzle_a);
  auto scale_recipe_b_enum = convert_int_to_enum<ScalingType>(scale_recipe_b);
  auto swizzle_b_enum = convert_int_to_enum<SwizzleType>(swizzle_b);

  // CPU does not support swizzle, so return false.
  TORCH_CHECK_VALUE(
      swizzle_a_enum[0] == at::blas::SwizzleType::NO_SWIZZLE &&
          swizzle_b_enum[0] == at::blas::SwizzleType::NO_SWIZZLE,
      "CPU does not support swizzle.");

  // at this point we can start working out what we want to be doing
  // Try to do as few steps as possible.
  // NOTE: support is deliberately sparse, can explicitly enumerate all
  // combinations allowed. Do this via a list of defined (name, acceptance,
  // concrete_impl) tuples.
  bool found_impl = false;
  ScaledGemmImplementation gemm_impl = ScaledGemmImplementation::NONE;

  for (const auto& fn_entry : scale_kernel_dispatch) {
    const auto [name, accept_fn, scaled_gemm_impl] = fn_entry;
    const bool ok = accept_fn(
        mat_a.scalar_type(),
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

  if (!found_impl) {
    invalid_scaling_config(mat_a, mat_b, scale_a[0], scale_b[0]);
  }

  at::native::resize_output(out, {mat_a.size(0), mat_b.size(1)});

  auto bias_ = bias.value_or(Tensor());

  if (gemm_impl == at::scaled::ScaledGemmImplementation::TENSORWISE_TENSORWISE ||
      gemm_impl == at::scaled::ScaledGemmImplementation::ROWWISE_ROWWISE) {
    _scaled_mm_out_cpu_emulated(
      mat_a,
      mat_b,
      scale_a[0],
      scale_b[0],
      bias,
      std::nullopt,
      out_dtype_,
      use_fast_accum,
      gemm_impl,
      out);
  } else {
    TORCH_CHECK_VALUE(false, "Invalid state - found an implementation, but not really");
  }

  return out;
}

Tensor _scaled_mm_cpu_v2(
    const Tensor& mat_a,
    const Tensor& mat_b,
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

  return _scaled_mm_cpu_v2_out(
    mat_a,
    mat_b,
    scale_a,
    scale_recipe_a,
    swizzle_a,
    scale_b,
    scale_recipe_b,
    swizzle_b,
    bias,
    out_dtype,
    contraction_dim,
    use_fast_accum,
    out);
}

// TODO(vasiliy, future PR): figure out why we need to declare this function, when
// other functions that live in ATen/native/*.cpp without declarations
// or headers work just fine.
Tensor _grouped_mm(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype);

Tensor _grouped_mm(const Tensor& mat_a, const Tensor& mat_b,
const std::optional<at::Tensor>& offs,
const std::optional<at::Tensor>& bias,
std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
  _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  return out;
}

}  // namespace at::native

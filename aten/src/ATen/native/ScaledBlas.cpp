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
#if !defined(__s390x__) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif

#if defined(_WIN32) && defined(USE_ROCM)
#include <windows.h>
#include <torch/library.h>
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
#include <ATen/ops/mul.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/_scaled_mm_v2.h>
#endif

namespace at::native {

static Tensor&
_scaled_mm_out_cpu_emulated(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  TORCH_INTERNAL_ASSERT((scale_a.numel() == 1 && scale_b.numel() == 1), "Now _scaled_mm only supports per-tensor scaling for CPU backend.");
  TORCH_CHECK(
      !scale_result ||
          (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
      "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());

  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());

  auto mat1_c = mat1.contiguous();
  auto mat2_c = mat2.contiguous();
  IntArrayRef mat1_sizes = mat1_c.sizes();
  IntArrayRef mat2_sizes = mat2_c.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  float input_scale = scale_a.item<float>();
  float weight_scale = scale_b.item<float>();
  float output_scale = 1.0f;
  if (scale_result.has_value() &&
      (*out_dtype == ScalarType::Float8_e4m3fn ||
       *out_dtype == ScalarType::Float8_e5m2)) {
    output_scale = scale_result.value().item<float>();
  }
  auto fp32_mat1 = at::mul(mat1.to(kFloat), input_scale);
  auto fp32_mat2 = at::mul(mat2_c.to(kFloat), weight_scale);
  auto out_tmp = at::matmul(fp32_mat1, fp32_mat2);
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
_scaled_mm_out_cpu(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
#if AT_MKLDNN_ENABLED() && !defined(__powerpc__)
  if (at::globalContext().userEnabledMkldnn()) {
    bool mixed_dtype = mat1.scalar_type() != mat2.scalar_type();
    if ((!mixed_dtype && cpuinfo_has_x86_amx_int8()) ||
        (mixed_dtype && cpuinfo_has_x86_amx_fp16())) {
      return mkldnn_scaled_mm(
          mat1,
          mat2,
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
  return _scaled_mm_out_cpu_emulated(mat1, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
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

// Windows ROCm: C-bridge for ABI compatibility between MSVC and Clang DLLs
Tensor _scaled_mm_v2_composite(
    const Tensor& mat_a, const Tensor& mat_b,
    ArrayRef<Tensor> scale_a, IntArrayRef scale_recipe_a, IntArrayRef swizzle_a,
    ArrayRef<Tensor> scale_b, IntArrayRef scale_recipe_b, IntArrayRef swizzle_b,
    const std::optional<Tensor>& bias,
    const std::optional<c10::ScalarType> out_dtype,
    IntArrayRef contraction_dim, bool use_fast_accum);

#if defined(_WIN32)
#include <windows.h>

using ScaledMmV2CBridge = Tensor (*)(
    const Tensor&, const Tensor&,
    const Tensor*, int64_t, const int64_t*, int64_t, const int64_t*, int64_t,
    const Tensor*, int64_t, const int64_t*, int64_t, const int64_t*, int64_t,
    const Tensor*, const c10::ScalarType*, const int64_t*, int64_t, bool);

static ScaledMmV2CBridge get_scaled_mm_v2_c_bridge() {
  static ScaledMmV2CBridge fn = nullptr;
  static bool tried = false;
  if (fn || tried) return fn;
  tried = true;
  HMODULE m = GetModuleHandleA("torch_hip.dll");
  if (!m) return nullptr;
  fn = reinterpret_cast<ScaledMmV2CBridge>(
      GetProcAddress(m, "_scaled_mm_cuda_v2_c_bridge"));
  return fn;
}
#endif

Tensor _scaled_mm_v2_composite(
    const Tensor& mat_a, const Tensor& mat_b,
    ArrayRef<Tensor> scale_a, IntArrayRef scale_recipe_a, IntArrayRef swizzle_a,
    ArrayRef<Tensor> scale_b, IntArrayRef scale_recipe_b, IntArrayRef swizzle_b,
    const std::optional<Tensor>& bias,
    const std::optional<c10::ScalarType> out_dtype,
    IntArrayRef contraction_dim, bool use_fast_accum) {

  TORCH_CHECK(mat_a.is_cuda() || mat_a.is_hip(),
      "_scaled_mm_v2 requires CUDA/HIP tensors");

#if defined(_WIN32)
  auto c_bridge = get_scaled_mm_v2_c_bridge();
  if (c_bridge) {
    c10::ScalarType dtype_val = out_dtype.value_or(c10::ScalarType::BFloat16);
    return c_bridge(
        mat_a, mat_b,
        scale_a.data(), static_cast<int64_t>(scale_a.size()),
        scale_recipe_a.data(), static_cast<int64_t>(scale_recipe_a.size()),
        swizzle_a.data(), static_cast<int64_t>(swizzle_a.size()),
        scale_b.data(), static_cast<int64_t>(scale_b.size()),
        scale_recipe_b.data(), static_cast<int64_t>(scale_recipe_b.size()),
        swizzle_b.data(), static_cast<int64_t>(swizzle_b.size()),
        bias ? &*bias : nullptr,
        out_dtype ? &dtype_val : nullptr,
        contraction_dim.data(), static_cast<int64_t>(contraction_dim.size()),
        use_fast_accum);
  }
#endif

  // Non-Windows-ROCm: use regular dispatcher
  const auto dtype = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(dtype));
  at::_scaled_mm_v2_outf(mat_a, mat_b,
      scale_a, scale_recipe_a, swizzle_a,
      scale_b, scale_recipe_b, swizzle_b,
      bias, out_dtype, contraction_dim, use_fast_accum, out);
  return out;
}

}  // namespace at::native

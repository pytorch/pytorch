#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/GroupedMMUtils.h>
#include <ATen/native/xpu/GroupedMM.h>

namespace at::native {

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

  bool use_fast_path =
      (mat_a.dtype() == at::kBFloat16 && mat_b.dtype() == at::kBFloat16 &&
       out_dtype_ == at::kBFloat16 &&
       at::native::xpu::is_grouped_mm_available());
  if (use_fast_path) {
    at::native::xpu::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  } else {
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}

} // namespace at::native

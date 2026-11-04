#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_nnpack_available_native.h>
#include <ATen/ops/_nnpack_spatial_convolution_native.h>
#endif

// NNPACK has been removed; these ops are kept so that existing callers get a clear error.
namespace at::native {

at::Tensor _nnpack_spatial_convolution(
    const Tensor& input,
    const Tensor& weight, const std::optional<Tensor>& bias_opt,
    const IntArrayRef padding,
    const IntArrayRef stride) {
  TORCH_CHECK(false, "_nnpack_spatial_convolution: NNPACK support has been removed from PyTorch");
}

bool _nnpack_available() {
  return false;
}

} // namespace at::native

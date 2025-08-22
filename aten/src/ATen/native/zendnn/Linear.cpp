#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/zendnn/Linear_utils.hpp>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zendnn_linear_native.h>
#endif

#if !AT_ZENDNN_ENABLED()
namespace at::native {
at::Tensor zendnn_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(false, "zendnn_linear: ATen not compiled with ZenDNN support");
}
} // namespace at::native

#else // !AT_ZENDNN_ENABLED()

namespace at::native {

at::Tensor zendnn_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  at::Tensor result;
  return result;
}
} // namespace at::native

#endif // !AT_ZENDNN_ENABLED()

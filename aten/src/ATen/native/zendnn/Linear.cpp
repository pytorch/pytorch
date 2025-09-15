#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/zendnn/Linear_utils.hpp>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zendnn_linear_unary_native.h>
#endif

#if !AT_ZENDNN_ENABLED()
namespace at::native {
at::Tensor zendnn_linear_unary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op) {
  TORCH_CHECK(
      false, "zendnn_linear_unary: ATen is not compiled with ZenDNN support");
}
} // namespace at::native

#else // !AT_ZENDNN_ENABLED()

namespace at::native {

at::Tensor zendnn_linear_unary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op) {
  at::Tensor result;
  return result;
}
} // namespace at::native

#endif // !AT_ZENDNN_ENABLED()

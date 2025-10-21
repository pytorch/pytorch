#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/zendnn/Linear_utils.hpp>
#include <string_view>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
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
using namespace zendnnl::interface;

inline void zendnn_linear_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& result,
    const std::string_view& post_op_id,
    bool is_weight_prepacked) {
  // Get appropriately processed tensors (2D input, transposed weight, 2D
  // result)
  check_args_for_linear(input, weight);
  data_type_t datatype = get_zendnn_dtype(input);
  auto input_2d = get_2d_view(input);
  auto weight_transposed = weight.t();
  auto result_2d = result.view(get_2d_size_for_tensor(result));
  check_tensor_dtypes_for_linear(input_2d, weight_transposed, bias, result_2d);
  check_tensor_sizes_for_linear(input_2d, weight_transposed, bias, result_2d);
  // declare linear tensors
  matmul_context_t matmul_context;
  tensor_t input_tensor, weight_tensor, output_tensor, bias_tensor;
  create_zendnn_tensor(input_2d, input_tensor, "matmul_input", datatype);
  create_zendnn_tensor(
      weight_transposed,
      weight_tensor,
      "weights",
      datatype,
      is_weight_prepacked);
  create_zendnn_tensor(result_2d, output_tensor, "matmul_output", datatype);
  if (bias.defined()) {
    // adds dimension at dim=0 -> [1, n]
    auto bias_unsqueezed = bias.unsqueeze(0);
    create_zendnn_tensor(bias_unsqueezed, bias_tensor, "bias", datatype);
    set_linear_context_attributes(
        matmul_context, weight_tensor, post_op_id, bias_tensor);
  } else {
    set_linear_context_attributes(matmul_context, weight_tensor, post_op_id);
  }
  matmul_context.create();
  // define matmul operator
  matmul_operator_t matmul_operator;
  matmul_operator.set_name("matmul_operator")
      .set_context(matmul_context)
      .create();
  TORCH_CHECK(
      matmul_operator.check(),
      "operator ",
      matmul_operator.get_name(),
      " creation failed.");
  matmul_operator.set_input("matmul_input", input_tensor)
      .set_output("matmul_output", output_tensor);
  matmul_operator.execute();
}

at::Tensor zendnn_linear_unary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_t = *bias_maybe_owned;
  // Create output tensor with appropriate size and strides
  at::Tensor result = create_linear_output_tensor(input, weight);
  // Perform ZENDNN linear operation
  zendnn_linear_impl(
      input, weight, bias_t, result, post_op, is_weight_prepacked);
  return result;
}
} // namespace at::native

#endif // !AT_ZENDNN_ENABLED()

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
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
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view zendnn_op_name) {
  TORCH_CHECK(false, "zendnn_linear: ATen is not compiled with ZenDNN support");
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
    const std::vector<int64_t>& post_op_ids,
    const std::vector<at::Tensor>& post_op_buffers,
    bool is_weight_prepacked,
    std::string_view zendnn_op_name) {
  data_type_t datatype = data_type_t::f32;
  if (input.scalar_type() == at::ScalarType::BFloat16) {
    datatype = data_type_t::bf16;
  }
  status_t status;
  TORCH_CHECK(
      (bias.dim() == 1 && input.dim() == 2 && weight.dim() == 2),
      "unsupported dims for bias, input and weight");
  check_valid_dtypes_for_matmul(input, weight, bias, result, {});
  check_valid_sizes_for_matmul(input, weight, bias, result, {});
  tensor_t weight_tensor = tensor_t();
  set_zendnn_tensor_attributes(
      weight, weight_tensor, "weights", datatype, is_weight_prepacked);
  weight_tensor.create();
  TORCH_CHECK(
      weight_tensor.check(),
      "tensor creation of ",
      weight_tensor.get_name(),
      " failed.");
  tensor_t input_tensor = tensor_t();
  set_zendnn_tensor_attributes(input, input_tensor, "matmul_input", datatype);
  input_tensor.create();
  TORCH_CHECK(
      input_tensor.check(),
      "tensor creation of ",
      input_tensor.get_name(),
      " failed.");

  // define matmul context
  auto matmul_context = matmul_context_t();
  int num_post_ops = 0;
  if (bias.defined()) {
    tensor_t bias_tensor = tensor_t();
    set_zendnn_tensor_attributes(bias, bias_tensor, "bias", datatype);
    bias_tensor.create();
    TORCH_CHECK(
        bias_tensor.check(),
        "tensor creation of ",
        bias_tensor.get_name(),
        " failed.");
    set_matmul_context_attributes(
        matmul_context, weight_tensor, post_op_ids, num_post_ops, bias_tensor);
  } else {
    set_matmul_context_attributes(
        matmul_context, weight_tensor, post_op_ids, num_post_ops);
  }
  matmul_context.create();

  // define matmul operator
  auto matmul_operator = matmul_operator_t()
                             .set_name("matmul_operator")
                             .set_context(matmul_context)
                             .create();
  TORCH_CHECK(
      matmul_operator.check(),
      "operator ",
      matmul_operator.get_name(),
      " creation failed.");
  // output tensor
  tensor_t output_tensor = tensor_t();
  set_zendnn_tensor_attributes(
      result, output_tensor, "matmul_output", datatype);
  output_tensor.create();
  TORCH_CHECK(
      output_tensor.check(),
      "tensor creation of ",
      output_tensor.get_name(),
      " failed.");
  status = matmul_operator.set_input("matmul_input", input_tensor)
               .set_output("matmul_output", output_tensor)
               .execute();
  if (status == status_t::success) {
    LOG(INFO) << "operator " << matmul_operator.get_name()
              << " execution successful for linear." << std::endl;
  } else {
    LOG(INFO) << "operator " << matmul_operator.get_name()
              << " execution failed for linear." << std::endl;
  }
}

at::Tensor zendnn_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view zendnn_op_name) {
  // Reshape input tensor to 2D view, handling both contiguous and
  // non-contiguous cases
  auto input_2d_view = input.is_contiguous()
      ? input.view(get_2d_size_for_tensor(input))
      : input.contiguous().view(get_2d_size_for_tensor(input));

  if (!weight.is_contiguous() && is_weight_prepacked) {
    TORCH_CHECK(
        false, "Prepacked weight tensor must be contiguous for zendnn_linear.");
  }
  // Transpose weight matrix for matrix multiplication
  auto weight_transposed = weight.t();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_t = *bias_maybe_owned;
  auto output_size = get_matmul_output_sizes(input, weight_transposed);
  auto output_strides = get_matmul_and_linear_output_strides(output_size);

  // Create output tensor with appropriate size and strides
  at::Tensor result = at::detail::empty_strided_cpu(
      output_size, output_strides, input.options());

  // Reshape output tensor to 2D view
  at::Tensor result_2d_view = result.view(get_2d_size_for_tensor(result));

  // Initialize post-operation containers
  std::vector<at::Tensor> post_op_buffers = {};
  std::vector<int64_t> post_op_ids = {};

  // Perform ZENDNN linear operation
  zendnn_linear_impl(
      input_2d_view,
      weight_transposed,
      bias_t,
      result_2d_view,
      post_op_ids,
      post_op_buffers,
      is_weight_prepacked,
      zendnn_op_name);

  return result;
}

} // namespace at::native
#endif // !AT_ZENDNN_ENABLED()

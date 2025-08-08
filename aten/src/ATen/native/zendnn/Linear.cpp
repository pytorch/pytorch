#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/zendnn/Linear_utils.hpp>
#include <ATen/native/zendnn/ZenDNN_utils.hpp>
#include <string_view>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zendnn_linear_native.h>
#include <ATen/ops/zendnn_linear_unary_binary_native.h>
#endif

#if !AT_ZENDNN_ENABLED()
namespace at::native {
at::Tensor zendnn_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op,
    std::string_view zendnn_op_name) {
  TORCH_CHECK(false, "zendnn_linear: ATen is not compiled with ZenDNN support");
}

at::Tensor zendnn_linear_unary_binary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& binary_input,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op_1,
    std::string_view post_op_2,
    std::string_view zendnn_op_name) {
  TORCH_CHECK(
      false,
      "zendnn_linear_unary_binary: ATen not compiled with ZenDNN support");
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
    const std::vector<std::string_view>& post_op_ids,
    const std::vector<at::Tensor>& post_op_buffers,
    bool is_weight_prepacked,
    std::string_view zendnn_op_name) {
  data_type_t datatype = data_type_t::f32;
  if (input.scalar_type() == at::ScalarType::BFloat16) {
    datatype = data_type_t::bf16;
  }
  TORCH_CHECK(
      (bias.dim() == 1 && input.dim() == 2 && weight.dim() == 2),
      "unsupported dims for bias, input and weight");
  check_valid_dtypes_for_matmul(input, weight, bias, result, {});
  check_valid_sizes_for_matmul(input, weight, bias, result, {});

  tensor_t input_tensor = tensor_t();
  set_zendnn_tensor_attributes(input, input_tensor, "matmul_input", datatype);
  input_tensor.create();
  TORCH_CHECK(
      input_tensor.check(),
      "tensor creation of ",
      input_tensor.get_name(),
      " failed.");

  tensor_t weight_tensor = tensor_t();
  set_zendnn_tensor_attributes(
      weight, weight_tensor, "weights", datatype, is_weight_prepacked);
  weight_tensor.create();
  TORCH_CHECK(
      weight_tensor.check(),
      "tensor creation of ",
      weight_tensor.get_name(),
      " failed.");

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

  // define matmul context
  auto matmul_context = matmul_context_t();
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
        matmul_context, weight_tensor, post_op_ids, bias_tensor);
  } else {
    set_matmul_context_attributes(matmul_context, weight_tensor, post_op_ids);
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
  matmul_operator.set_input("matmul_input", input_tensor)
      .set_output("matmul_output", output_tensor);

  if (post_op_buffers.size() > 0) {
    int empty_post_op_id = 0;
    for (size_t i = 0; i < post_op_ids.size(); i++) {
      if (post_op_ids[i] == "no_post_op_!")
        empty_post_op_id++;
      if (post_op_ids[i] == "mul") {
        tensor_t binary_tensor = tensor_t();
        set_zendnn_tensor_attributes(
            post_op_buffers[i], binary_tensor, "binary_input", datatype);
        binary_tensor.create();
        matmul_operator.set_input(
            matmul_context.get_post_op(i - empty_post_op_id)
                .binary_mul_params.tensor_name,
            binary_tensor);
      } else if (post_op_ids[i] == "add") {
        tensor_t binary_tensor = tensor_t();
        set_zendnn_tensor_attributes(
            post_op_buffers[i], binary_tensor, "binary_input", datatype);
        binary_tensor.create();
        matmul_operator.set_input(
            matmul_context.get_post_op(i - empty_post_op_id)
                .binary_add_params.tensor_name,
            binary_tensor);
      }
    }
  }
  matmul_operator.execute();
}

at::Tensor zendnn_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op,
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

  // Convert post_op_ids string to vector of string_views
  std::vector<std::string_view> post_op_ids_vec;
  if (!post_op.empty()) {
    // For now, treat as a single post-op ID
    // TODO: Parse comma-separated or JSON format if needed
    post_op_ids_vec.push_back(post_op);
  } else {
    post_op_ids_vec.emplace_back("no_post_op_!");
  }

  // Perform ZENDNN linear operation
  zendnn_linear_impl(
      input_2d_view,
      weight_transposed,
      bias_t,
      result_2d_view,
      post_op_ids_vec,
      post_op_buffers,
      is_weight_prepacked,
      zendnn_op_name);

  return result;
}

at::Tensor zendnn_linear_unary_binary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& binary_input,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op_1,
    std::string_view post_op_2,
    std::string_view zendnn_op_name) {
  // Reshape input tensor to 2D view, handling both contiguous and
  // non-contiguous cases
  auto input_2d_view = input.is_contiguous()
      ? input.view(get_2d_size_for_tensor(input))
      : input.contiguous().view(get_2d_size_for_tensor(input));

  // Transpose weight matrix for matrix multiplication
  auto weight_transposed = weight.t();
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_t = *bias_maybe_owned;
  at::Tensor result;
  // Create output tensor with appropriate size and strides
  if (post_op_2 == "add") {
    result = at::detail::empty_strided_cpu(
        binary_input.sizes(), binary_input.strides(), binary_input.options());
  } else {
    auto output_size = get_matmul_output_sizes(input, weight_transposed);
    auto output_strides = get_matmul_and_linear_output_strides(output_size);
    result = at::detail::empty_strided_cpu(
        output_size, output_strides, input.options());
  }
  auto binary_input_2d_view = binary_input.is_contiguous()
      ? binary_input.view(get_2d_size_for_tensor(binary_input))
      : binary_input.contiguous().view(get_2d_size_for_tensor(binary_input));
  // Reshape output tensor to 2D view
  at::Tensor result_2d_view = result.view(get_2d_size_for_tensor(result));

  // Initialize post-operation containers
  std::vector<std::string_view> post_op_ids;
  if (post_op_1.empty()) {
    post_op_ids.emplace_back("no_post_op_!");
  } else {
    post_op_ids.emplace_back(post_op_1);
  }
  if (!post_op_2.empty()) {
    post_op_ids.emplace_back(post_op_2);
  }
  std::vector<at::Tensor> post_op_buffers;
  // Push an empty tensor for first post_op id
  const at::Tensor empty_tensor;
  post_op_buffers.emplace_back(empty_tensor);

  post_op_buffers.emplace_back(binary_input_2d_view);
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

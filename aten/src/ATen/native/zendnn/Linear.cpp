#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/zendnn/Linear_utils.hpp>
#include <string_view>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/zendnn_linear_binary_binary_native.h>
#include <ATen/ops/zendnn_linear_unary_binary_native.h>
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

at::Tensor zendnn_linear_unary_binary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& binary_input,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op_1,
    std::string_view post_op_2) {
  TORCH_CHECK(
      false,
      "zendnn_linear_unary_binary: ATen is not compiled with ZenDNN support");
}

at::Tensor zendnn_linear_binary_binary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& binary_input_1,
    const at::Tensor& binary_input_2,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op_1,
    std::string_view post_op_2) {
  TORCH_CHECK(
      false,
      "zendnn_linear_binary_binary: ATen is not compiled with ZenDNN support");
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
    bool is_weight_prepacked) {
  // Get appropriately processed tensors (2D input, transposed weight, 2D
  // result)
  check_args_for_linear(input, weight);
  auto input_2d = get_2d_view(input);
  auto weight_transposed = weight.t();
  auto result_2d = result.view(get_2d_size_for_tensor(result));
  check_tensor_dtypes_for_linear(
      input_2d, weight_transposed, bias, result_2d, post_op_buffers);
  check_tensor_sizes_for_linear(
      input_2d, weight_transposed, bias, result_2d, post_op_buffers);

  // Use direct matmul
  const int64_t M = input_2d.size(0);
  const int64_t N = weight_transposed.size(1);
  const int64_t K = input_2d.size(1);

  // check if tensor is transposed
  bool transa = is_transposed(input_2d);
  bool transb = is_transposed(weight_transposed);
  // make a copy of tensor when tensor is neither contiguous nor transposed
  input_2d =
      (transa || input_2d.is_contiguous()) ? input_2d : input_2d.contiguous();
  weight_transposed = (transb || weight_transposed.is_contiguous())
      ? weight_transposed
      : weight_transposed.contiguous();
  auto strideA = input_2d.strides();
  auto strideB = weight_transposed.strides();
  auto strideC = result_2d.strides();

  const int64_t lda = transa ? strideA[1] : strideA[0];
  const int64_t ldb = transb ? strideB[1] : strideB[0];
  const int64_t ldc = strideC[0];
  zendnnl::lowoha::data_types matmul_dtype;
  matmul_dtype.src = get_zendnn_dtype(input_2d);
  matmul_dtype.wei = get_zendnn_dtype(weight_transposed);
  matmul_dtype.dst = get_zendnn_dtype(result_2d);
  matmul_dtype.bias =
      (bias.defined()) ? get_zendnn_dtype(bias) : data_type_t::none;
  std::vector<zendnnl::lowoha::postop> post_op;
  if (post_op_ids.size() > 0) {
    // Set post-op parameters
    for (size_t i = 0; i < post_op_ids.size(); i++) {
      zendnnl::lowoha::postop op1;
      auto it = post_op_map.find(post_op_ids[i]);
      if (it == post_op_map.end()) {
        continue;
      } else {
        op1.po_type = it->second;
      }
      if ((post_op_ids[i] == "mul" || post_op_ids[i] == "add")) {
        auto sizes = post_op_buffers[i].sizes();
        op1.dims.assign(sizes.begin(), sizes.end());
        op1.dtype = get_zendnn_dtype(post_op_buffers[i]);
        op1.buff = post_op_buffers[i].data_ptr();
      }
      post_op.push_back(op1);
    }
  }
  zendnnl::lowoha::lowoha_params params;
  params.dtypes = matmul_dtype;
  params.postop_ = std::move(post_op);
  params.mem_format_a = 'n';
  params.mem_format_b = is_weight_prepacked ? 'r' : 'n';

  // Execute Linear directly for LoA path
  matmul_direct(
      'r',
      transa,
      transb,
      M,
      N,
      K,
      1.0f, /*alpha*/
      input_2d.data_ptr(),
      lda,
      weight_transposed.data_ptr(),
      ldb,
      (bias.defined()) ? bias.data_ptr() : nullptr,
      0.0f, /*beta*/
      result_2d.data_ptr(),
      ldc,
      params,
      1, /*for matmul*/
      1 /*for matmul*/
  );
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
  std::vector<std::string_view> post_op_ids =
      std::vector<std::string_view>{post_op};
  // Perform ZENDNN linear operation
  zendnn_linear_impl(
      input,
      weight,
      bias_t,
      result,
      post_op_ids,
      {} /* post_op_buffers */,
      is_weight_prepacked);
  return result;
}

at::Tensor zendnn_linear_unary_binary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& binary_input,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op_1,
    std::string_view post_op_2) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_t = *bias_maybe_owned;
  // Create output tensor with appropriate size and strides
  at::Tensor result = create_linear_output_tensor(input, weight);
  // Initialize post-operation containers
  std::vector<std::string_view> post_op_ids =
      std::vector<std::string_view>{post_op_1, post_op_2};
  std::vector<at::Tensor> post_op_buffers;
  // Push an empty tensor for first post_op id as its always unary
  const at::Tensor empty_tensor;
  post_op_buffers.emplace_back(empty_tensor);
  post_op_buffers.emplace_back(get_2d_view(binary_input));
  // Perform ZENDNN linear operation
  zendnn_linear_impl(
      input,
      weight,
      bias_t,
      result,
      post_op_ids,
      post_op_buffers,
      is_weight_prepacked);
  return result;
}

at::Tensor zendnn_linear_binary_binary(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& binary_input_1,
    const at::Tensor& binary_input_2,
    const std::optional<at::Tensor>& bias,
    bool is_weight_prepacked,
    std::string_view post_op_1,
    std::string_view post_op_2) {
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias);
  const at::Tensor& bias_t = *bias_maybe_owned;

  at::Tensor result = create_linear_output_tensor(input, weight);
  // Initialize post-operation containers
  std::vector<std::string_view> post_op_ids =
      std::vector<std::string_view>{post_op_1, post_op_2};
  std::vector<at::Tensor> post_op_buffers = std::vector<at::Tensor>{
      get_2d_view(binary_input_1), get_2d_view(binary_input_2)};

  // Perform ZENDNN linear operation
  zendnn_linear_impl(
      input,
      weight,
      bias_t,
      result,
      post_op_ids,
      post_op_buffers,
      is_weight_prepacked);

  return result;
}

} // namespace at::native

#endif // !AT_ZENDNN_ENABLED()

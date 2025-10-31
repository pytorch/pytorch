#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/zendnn/ZenDNN_utils.hpp>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/as_strided.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zendnn_weight_prepack_for_linear_native.h>
#endif

#if !AT_ZENDNN_ENABLED()
namespace at::native {
at::Tensor zendnn_weight_prepack_for_linear(const at::Tensor& weight) {
  TORCH_CHECK(
      false,
      "zendnn_weight_prepack_for_linear: ATen is not compiled with ZenDNN support");
}
} // namespace at::native
#else // !AT_ZENDNN_ENABLED()
namespace at::native {
using namespace zendnnl::interface;
at::Tensor zendnn_weight_prepack_for_linear(const at::Tensor& weight) {
  TORCH_CHECK(
      weight.dim() == 2,
      "Weight tensor must be 2D for linear layer prepacking, got ",
      weight.dim(),
      "D tensor.");
  TORCH_CHECK(
      weight.scalar_type() == c10::ScalarType::Float ||
          weight.scalar_type() == c10::ScalarType::BFloat16,
      "Currently weight prepacking only supports float32 or bfloat16 dtype for weight tensor");
  data_type_t datatype = get_zendnn_dtype(weight);
  // Linear op internally works on transposed weight tensor, so to
  // prepack the weight we need to use transposed weight.
  auto reorder_input = weight.t();
  tensor_t zen_reorder_input;
  create_zendnn_tensor(
      reorder_input, zen_reorder_input, "reorder_input", datatype);
  // Currently, ZenDNN only supports blocked layout with AOCL kernels.
  auto context = reorder_context_t().set_algo_format("aocl").create();
  auto reorder_op =
      reorder_operator_t().set_name("reorder_op").set_context(context).create();
  // Check if reorder operation creation is successful.
  TORCH_CHECK(
      reorder_op.check(),
      "operator ",
      reorder_op.get_name(),
      " creation failed.");
  reorder_op.set_input("reorder_input", zen_reorder_input);
  size_t reorder_bytes = reorder_op.get_reorder_size();
  int64_t num_elements = reorder_bytes / weight.element_size();
  // Create 1d tensor to hold the reordered weights with
  // a stride of 1 to ensure contiguous memory layout.
  at::Tensor reorder_output = at::detail::empty_strided_cpu(
      /*size*/ {num_elements}, /*stride*/ {1}, weight.options());
  tensor_t zen_reorder_output;
  std::vector<long unsigned int> reorder_output_sizes(
      reorder_input.sizes().begin(), reorder_input.sizes().end());
  void* reorder_output_ptr = reorder_output.data_ptr();
  zen_reorder_output.set_name("reorder_output")
      .set_size(reorder_output_sizes)
      .set_data_type(datatype)
      .set_storage(reorder_output_ptr, reorder_output.nbytes());
  if (is_tensor_2d_and_transposed(reorder_input)) {
    zen_reorder_output.set_order("ba");
  }
  zen_reorder_output.set_layout(tensor_layout_t::blocked);
  zen_reorder_output.create();
  // Check if reorder output tensor creation is successful.
  TORCH_CHECK(
      zen_reorder_output.check(),
      "tensor creation of ",
      zen_reorder_output.get_name(),
      " failed.");
  reorder_op.set_output("reorder_output", zen_reorder_output);
  reorder_op.execute();
  return at::as_strided(reorder_output, weight.sizes(), weight.strides());
}
} // namespace at::native
#endif // !AT_ZENDNN_ENABLED()

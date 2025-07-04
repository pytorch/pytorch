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
at::Tensor zendnn_weight_prepack_for_linear(
    const at::Tensor& weight,
    bool treat_tensor_as_transposed,
    std::string_view zendnn_op_name) {
  TORCH_CHECK(
      false,
      "zendnn_weight_prepack_for_linear: ATen is not compiled with ZenDNN support");
}
} // namespace at::native
#else // !AT_ZENDNN_ENABLED()
namespace at::native {
using namespace zendnnl::interface;

at::Tensor zendnn_weight_prepack_for_linear(
    const at::Tensor& weight,
    bool treat_tensor_as_transposed,
    std::string_view zendnn_op_name) {
  TORCH_CHECK(
      weight.dim() == 2,
      "Weight tensor must be 2D for linear layer prepacking, got ",
      weight.dim(),
      "D tensor.");

  data_type_t data_type;
  switch (weight.scalar_type()) {
    case at::ScalarType::Float:
      data_type = data_type_t::f32;
      break;
    case at::ScalarType::BFloat16:
      data_type = data_type_t::bf16;
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported data type for weight tensor to prepack, only Float and BFloat16 "
          "are supported.");
  }

  auto reorder_input = treat_tensor_as_transposed ? weight.t() : weight;

  status_t status;

  tensor_t zen_reorder_input = tensor_t();
  set_zendnn_tensor_attributes(
      reorder_input, zen_reorder_input, "reorder_input", data_type);
  zen_reorder_input.create();
  TORCH_CHECK(
      zen_reorder_input.check(),
      "tensor creation of ",
      zen_reorder_input.get_name(),
      " failed.");

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

  at::Tensor reorder_output =
      at::detail::empty_strided_cpu({num_elements}, {1}, weight.options());

  tensor_t zen_reorder_output = tensor_t();

  std::vector<long unsigned int> reorder_output_sizes;
  auto reorder_input_sizes = reorder_input.sizes();
  for (auto val : reorder_input_sizes) {
    reorder_output_sizes.emplace_back(static_cast<long unsigned int>(val));
  }

  void* reorder_output_ptr = reorder_output.data_ptr();

  zen_reorder_output.set_name("reorder_output")
      .set_size(reorder_output_sizes)
      .set_data_type(data_type)
      .set_storage(reorder_output_ptr, reorder_output.nbytes());
  if (treat_tensor_as_transposed) {
    zen_reorder_output.set_order("ba");
  }
  zen_reorder_output.set_layout(tensor_layout_t::blocked);

  zen_reorder_output.create();

  TORCH_CHECK(
      zen_reorder_output.check(),
      "tensor creation of ",
      zen_reorder_output.get_name(),
      " failed.");

  reorder_op.set_output("reorder_output", zen_reorder_output);

  status = reorder_op.execute();

  if (status == status_t::success) {
    LOG(INFO) << "operator " << reorder_op.get_name()
              << " execution successful.";
  } else {
    LOG(INFO) << "operator " << reorder_op.get_name() << " execution failed.";
  }

  return at::as_strided(reorder_output, weight.sizes(), weight.strides());
}

} // namespace at::native
#endif // !AT_ZENDNN_ENABLED()

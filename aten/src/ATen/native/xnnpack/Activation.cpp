#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/Engine.h>
#include <ATen/native/utils/Factory.h>

namespace at::native::xnnpack {


bool use_hardswish(
  const Tensor& input) {
  return xnnpack::available() &&
          (1 <= input.ndimension()) &&
          (input.device().is_cpu()) &&
          (kFloat == input.scalar_type()) &&
          !input.requires_grad() &&
           true;
}

static Tensor& hardswish_impl(Tensor& input, Tensor& output) {
  using namespace internal;
  // Create XNNPACK Subgraph
  xnn_subgraph_t subgraph_ptr = nullptr;
  xnn_status status = xnn_create_subgraph(
    /*external_value_ids=*/2,
    /*flags=*/0,
    &subgraph_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn create subgraph failed(", status,")!");
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
      subgraph_ptr, &xnn_delete_subgraph);
  uint32_t input_id = XNN_INVALID_VALUE_ID, output_id = XNN_INVALID_VALUE_ID;
  std::vector<size_t> input_output_shape(input.sizes().begin(), input.sizes().end());

  status = xnn_define_tensor_value(
    subgraph_ptr,
    xnn_datatype_fp32,
    input_output_shape.size(),
    input_output_shape.data(),
    nullptr,
    0,
    XNN_VALUE_FLAG_EXTERNAL_INPUT,
    &input_id
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "defining xnn input failed(", status,")!");

  status = xnn_define_tensor_value(
    subgraph_ptr,
    xnn_datatype_fp32,
    input_output_shape.size(),
    input_output_shape.data(),
    nullptr,
    1,
    XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
    &output_id
  );
  TORCH_CHECK(
      status == xnn_status_success,
      "defining xnn output failed(", status,")!");

  status = xnn_define_unary(
    subgraph_ptr,
    xnn_unary_hardswish,
    nullptr,
    input_id,
    output_id,
    0
  );

  // create runtime
  xnn_runtime_t runtime_ptr = nullptr;
  status = xnn_create_runtime_v2(subgraph_ptr, caffe2::pthreadpool_(), 0, &runtime_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn create runtime failed(", status,")!");
  TORCH_CHECK(
      runtime_ptr != nullptr,
      "xnn create runtime failed because runtime_ptr is null");
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime_ptr, &xnn_delete_runtime);

  std::array<xnn_external_value, 2> external = {
    xnn_external_value{input_id, input.data_ptr<float>()},
    xnn_external_value{output_id, output.data_ptr<float>()}};

  status = xnn_setup_runtime(
    runtime_ptr,
    external.size(),
    external.data());
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn setup runtime failed(", status,")!");
  status = xnn_invoke_runtime(runtime_ptr);
  TORCH_CHECK(
      status == xnn_status_success,
      "xnn invoke runtime failed(", status,")!");

  return output;
}

Tensor hardswish(const Tensor& input) {
  Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
    input, input.suggest_memory_format());

  Tensor output = mobile::empty_with_tail_padding(
    padded_input.sizes(),
    padded_input.options().dtype(),
    input.suggest_memory_format(),
    padded_input.opt_names());

  hardswish_impl(padded_input, output);
  return output.contiguous(input.suggest_memory_format());
}

Tensor& hardswish_(Tensor& input) {
  Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
    input, input.suggest_memory_format());

  // Don't need to allocate output if input is contiguous & already padded
  if (input.data_ptr() == padded_input.data_ptr()) {
    hardswish_impl(input, input);
    return input;
  } else {
    Tensor output = mobile::empty_with_tail_padding(
      padded_input.sizes(),
      padded_input.options().dtype(),
      input.suggest_memory_format(),
      padded_input.opt_names());
    hardswish_impl(padded_input, output);
    return input.copy_(output);
  }
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

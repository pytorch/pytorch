#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/utils/Factory.h>

namespace at {
namespace native {
namespace xnnpack {


bool use_hardswish(
  const Tensor& input) {
  return xnnpack::internal::available() &&
          (1 <= input.ndimension()) &&
          (input.device().is_cpu()) &&
          (kFloat == input.scalar_type()) &&
          !input.requires_grad() &&
           true;
}

Tensor& hardswish_impl(Tensor& input, Tensor& output) {
  using namespace internal;

  xnn_operator_t hardswish_op{};
  const auto channels = Layout::ActivationND::channel(input.sizes());
  const xnn_status create_status = xnn_create_hardswish_nc_f32(
    channels, // channels
    channels, // input stride
    channels, // output stride
    0, // flags
    &hardswish_op);

  TORCH_CHECK(
    xnn_status_success == create_status,
    "xnn_create_hardswish_nc_f32 failed!");

  Operator hardswish_scoped_op(hardswish_op);

  const xnn_status setup_status = xnn_setup_hardswish_nc_f32(
    hardswish_op,
    Layout::ActivationND::batch(input.sizes()),  // Batch
    input.data_ptr<float>(),
    output.data_ptr<float>(),
    caffe2::pthreadpool_());  // threadpool

  TORCH_CHECK(
    xnn_status_success == setup_status,
    "xnn_setup_hardswish_nc_f32 failed!");

  const xnn_status run_status = xnn_run_operator(
    hardswish_op,
    caffe2::pthreadpool_());  // threadpool

  TORCH_INTERNAL_ASSERT(
    xnn_status_success == run_status,
    "xnn_run_operator failed!");

  return output;
}

Tensor hardswish(const Tensor& input) {
  Tensor padded_input = mobile::allocate_padded_contiguous_if_needed(
    input, input.suggest_memory_format());

  Tensor output = mobile::empty_with_tail_padding(
    padded_input.sizes(),
    padded_input.options().dtype(),
    input.suggest_memory_format(),
    padded_input.names());

  hardswish_impl(padded_input, output);
  return output.contiguous(input.suggest_memory_format());
}

Tensor& hardswish_(Tensor& input) {
  using namespace internal;

  hardswish_impl(input, input);
  return input;
}

}
}
}

#endif /* USE_XNNPACK */

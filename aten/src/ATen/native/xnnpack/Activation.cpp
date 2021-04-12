#ifdef USE_XNNPACK

#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/utils/Factory.h>

namespace at {
namespace native {
namespace xnnpack {

Tensor hardswish(const Tensor& input) {
  using namespace internal;

  const Tensor padded_input =
      mobile::allocate_padded_contiguous_if_needed(
          input, input.suggest_memory_format());

  Tensor output = mobile::empty_with_tail_padding(
      padded_input.sizes(),
      padded_input.options().dtype(),
      input.suggest_memory_format(),
      padded_input.names());

  xnn_operator_t hardswish_op{};
  const auto channels = Layout::ActivationND::channel(padded_input.sizes());
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
    Layout::ActivationND::batch(padded_input.sizes()),  // Batch
    padded_input.data_ptr<float>(),
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

  return output.contiguous(input.suggest_memory_format());
}

}
}
}

#endif /* USE_XNNPACK */

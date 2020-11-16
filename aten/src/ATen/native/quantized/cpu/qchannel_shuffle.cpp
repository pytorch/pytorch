#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <c10/core/TensorOptions.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <algorithm>

namespace at {
namespace native {

#ifdef USE_PYTORCH_QNNPACK
namespace {
Tensor quantized_channel_shuffle_impl(
    const Tensor& self,
    int64_t groups) {

  TORCH_CHECK(
      groups > 0,
      "Number of groups to divide channels in must be positive.",
      " Value of groups:", groups);
  TORCH_CHECK(
      self.dim() == 4,
      "channel_shuffle expects 4D input, but got input with sizes ",
      self.sizes());
  TORCH_CHECK(
      self.scalar_type() == kQUInt8,
      "Quantized channel shuffle works only on uint8_t.",
      "But got:", self.scalar_type());
  const Tensor self_nhwc = self.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::native::empty_affine_quantized(
      self_nhwc.sizes(),
      at::device(kCPU).dtype(kQUInt8).memory_format(MemoryFormat::ChannelsLast),
      self_nhwc.q_scale(),
      self_nhwc.q_zero_point(),
      c10::nullopt
      );

  // Degenerate case of just copying.
  if (groups == 1) {
    qy.copy_(self_nhwc);
    return qy.contiguous(self.suggest_memory_format());
  }

  int64_t channels = self.size(1);
  TORCH_CHECK(channels > 0,
             "Number of channels must be positive, got:", channels);
  TORCH_CHECK((channels % groups) == 0,
             "Number of channels must be divisible gy groups. Got ",
             channels, " channels and ", groups, " groups.");

  initQNNPACK();

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_channel_shuffle_nc_x8(
      groups /* groups */,
      channels / groups /* group channels */,
      0 /* flags */,
      &qnnpack_operator);
  TORCH_INTERNAL_ASSERT(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK ChannelShuffle operator");

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_channel_shuffle_nc_x8(
      qnnpack_uniq_ptr.get(),
      self_nhwc.numel() / channels /* batch size */,
      (uint8_t*)self_nhwc.data_ptr<c10::quint8>() /* self data */,
      channels /* self stride */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* qy data */,
      channels /* qy stride */);
  TORCH_INTERNAL_ASSERT(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK ChannelShuffle operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK ChannelShuffle operator");

  return qy.contiguous(self.suggest_memory_format());
}
} // namespace
#endif

// at::native functions for the native_functions.yaml
Tensor channel_shuffle_quantized_cpu(
    const Tensor& self,
    int64_t groups) {
#ifdef USE_PYTORCH_QNNPACK
  return quantized_channel_shuffle_impl(self, groups);
#endif
  // If QNNPACK is not available then fall back to the
  // non quantized path.
  return at::native::channel_shuffle(self, groups);
}

// Keep the registry in the anonymous namespace.
namespace {
class QChannelShuffle final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, int64_t groups) {
    return channel_shuffle_quantized_cpu(qx, groups);
  }
};

} // namespace

} // namespace native
} // namespace at

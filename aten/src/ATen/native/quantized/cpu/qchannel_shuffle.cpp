#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <c10/core/TensorOptions.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>

#include <algorithm>

namespace at {
namespace native {
namespace {

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_channel_shuffle_impl(
    const Tensor& self,
    int64_t groups) {

  TORCH_CHECK(
      self.scalar_type() == kQUInt8,
      "Qnnpack channel shuffle works only on ",
      toString(c10::kQUInt8),
      " but got ", self.scalar_type());
  const Tensor self_nhwc = self.contiguous(MemoryFormat::ChannelsLast);
  Tensor qy = at::native::empty_affine_quantized(
      self_nhwc.sizes(),
      kQUInt8,
      c10::nullopt /* layout */,
      kCPU,
      c10::nullopt /* pin_memory */,
      self_nhwc.q_scale(),
      self_nhwc.q_zero_point(),
      MemoryFormat::ChannelsLast);

  initQNNPACK();

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  int64_t channels = self.size(1);
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
#endif

template<typename scalar_t>
Tensor quantized_channel_shuffle_impl(
    const Tensor& self,
    int64_t groups) {

  int64_t nbatch = self.size(0);
  int64_t channels = self.size(1);
  int64_t channels_per_group = channels / groups;
  int64_t image_size = self.numel() / nbatch / channels;

  if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    Tensor qy = at::_empty_affine_quantized(
        self.sizes(),
        self.options().memory_format(at::MemoryFormat::ChannelsLast),
        self.q_scale(),
        self.q_zero_point(),
        c10::nullopt);

    scalar_t* idata = static_cast<scalar_t*>(self.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());
    at::parallel_for(0, nbatch * image_size, 0, [&](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        utils::transpose<scalar_t>(
            groups /* M */,
            channels_per_group /* N */,
            idata + i * channels /* src */,
            channels_per_group /* ld_src */,
            odata + i * channels /* dst */,
            groups /* ld_dst */);
      }
    });
    return qy;
  } else {
    Tensor qy = at::_empty_affine_quantized(
        self.sizes(),
        self.options(),
        self.q_scale(),
        self.q_zero_point());
    auto self_contig = self.contiguous();

    scalar_t* idata = static_cast<scalar_t*>(self_contig.data_ptr());
    scalar_t* odata = static_cast<scalar_t*>(qy.data_ptr());
    at::parallel_for (0, nbatch * /* oc*g */channels, 0, [&](int64_t begin, int64_t end) {
      int64_t n{0}, oc{0}, g{0};
      data_index_init(begin, n, nbatch, oc, channels_per_group, g, groups);

      for (const auto i : c10::irange(begin, end)) {
        scalar_t* output_ptr = odata + i * image_size;
        scalar_t* input_ptr = idata + n * channels * image_size +
            g * channels_per_group * image_size + oc * image_size;
        memcpy(output_ptr, input_ptr, image_size * sizeof(scalar_t));

        // move on to next output index
        data_index_step(n, nbatch, oc, channels_per_group, g, groups);
      }
    });
    return qy;
  }
}

} // namespace

// at::native functions for the native_functions.yaml
Tensor channel_shuffle_quantized_cpu(
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

  // Degenerate case of just copying.
  if (groups == 1) {
    return self.clone();
  }

  int64_t channels = self.size(1);
  TORCH_CHECK(channels > 0,
             "Number of channels must be positive, got:", channels);
  TORCH_CHECK((channels % groups) == 0,
             "Number of channels must be divisible gy groups. Got ",
             channels, " channels and ", groups, " groups.");

#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK && self.scalar_type() == kQUInt8) {
    return qnnpack_channel_shuffle_impl(self, groups);
  }
#endif
  Tensor output;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "channel_shuffle_quantized_cpu", [&]() {
    output = quantized_channel_shuffle_impl<scalar_t>(self, groups);
  });
  return output;
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

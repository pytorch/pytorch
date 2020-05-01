#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qhardswish_stub);

namespace {

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_hardswish(const Tensor& qx, Tensor& qy) {
  TORCH_CHECK(qx.ndimension() > 0, "qnnpack_hardswish(): Got empty input tensor");
  initQNNPACK();

  size_t num_elems = qx.numel() / qx.size(0);
  const auto i_zero_point = qx.q_zero_point();
  const auto i_scale = qx.q_scale();
  const auto o_zero_point = qy.q_zero_point();
  const auto o_scale = qy.q_scale();

  pytorch_qnnp_operator_t hardswish_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_hardswish_nc_q8(
    num_elems, // channels
    i_zero_point,
    i_scale,
    o_zero_point,
    o_scale,
    std::numeric_limits<uint8_t>::min(), // output min
    std::numeric_limits<uint8_t>::max(), // output max
    0, // flags
    &hardswish_op);
  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Hardswish operator");

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_hardswish_nc_q8(
    hardswish_op,
    qx.size(0), // batch size
    (uint8_t*)qx.data_ptr<c10::quint8>(), // input data
    num_elems, // input stride
    (uint8_t*)qy.data_ptr<c10::quint8>(), // output data
    num_elems); // output stride
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Hardswish operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(hardswish_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Hardswish operator");
  return qy;
}
#endif // USE_PYTORCH_QNNPACK

} // namespace

Tensor quantized_hardswish(const Tensor& qx, double output_scale, int64_t output_zero_point) {
  Tensor qy = at::_empty_affine_quantized(
      qx.sizes(),
      at::device(kCPU)
        .dtype(qx.scalar_type()),
        // .memory_format(MemoryFormat::ChannelsLast),
      output_scale,
      output_zero_point,
      c10::nullopt);
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    Tensor qx_contig = qx.contiguous(qx.suggest_memory_format());
    TORCH_CHECK(qy.is_contiguous(), "qy must be contiguous");
    qnnpack_hardswish(qx_contig, qy);
    return qy;
  }
#endif  // USE_PYTORCH_QNNPACK
  qhardswish_stub(qx.device().type(), qx, qy);
  return qy;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl("hardswish", quantized_hardswish);
}

}}  // namespace at::native

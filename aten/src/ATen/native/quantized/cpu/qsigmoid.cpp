#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qsigmoid_stub);

#ifdef USE_PYTORCH_QNNPACK
// This ALWAYS outputs scale=1.0/256, dtype=quint8
// The zero_point is 0 for qint32 and quint8, but -128 for qint8.
Tensor qnnpack_sigmoid(Tensor input) {
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_sigmoid(): Got empty input tensor");

  Tensor qy;
  constexpr float output_scale = 1.0f / 256.0f;
  constexpr int32_t output_zero_point = 0;

  initQNNPACK();

  Tensor input_contig = input.contiguous();
  size_t num_elems = input_contig.numel() / input_contig.size(0);

  const auto zero_point = input_contig.q_zero_point();
  const auto scale = input_contig.q_scale();

  pytorch_qnnp_operator_t sigmoid_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_sigmoid_nc_q8(
    num_elems /* channels */,
    zero_point /* input zero point */,
    scale /* input scale */,
    output_zero_point /* output zero point */,
    output_scale /* output scale */,
    std::numeric_limits<uint8_t>::min() /* output min */,
    std::numeric_limits<uint8_t>::max() /* output max */,
    0 /* flags */,
    &sigmoid_op);
  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK sigmoid operator");
  qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    input.options(),
    output_scale,
    output_zero_point);

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_sigmoid_nc_q8(
    sigmoid_op,
    input_contig.size(0) /* batch size */,
    (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
    num_elems /* input stride */,
    (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
    num_elems /* output stride */);
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK sigmoid operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(sigmoid_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK sigmoid operator");
  return qy;
}
#endif  // USE_PYTORCH_QNNPACK

Tensor quantized_sigmoid(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    return qnnpack_sigmoid(qx);
  }
#endif  // USE_PYTORCH_QNNPACK
  Tensor qy;
  qsigmoid_stub(qx.device().type(), qx, qy);
  return qy;
}
}}  // namespace at::native

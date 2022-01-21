#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/tanh_native.h>
#endif

namespace at {
namespace native {

DEFINE_DISPATCH(qtanh_stub);

#ifdef USE_PYTORCH_QNNPACK
// This ALWAYS outputs scale=2.0/256, zp=128, dtype=quint8
Tensor qnnpack_tanh(Tensor input) {
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_tanh(): Got empty input tensor");
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_tanh(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));
  Tensor qy;
  constexpr float output_scale = 2.0f / 256.0f;
  constexpr int32_t output_zero_point = 128;

  initQNNPACK();

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }
  const auto zero_point = input_contig.q_zero_point();
  const auto scale = input_contig.q_scale();

  pytorch_qnnp_operator_t tanh_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_tanh_nc_q8(
    num_elems /* channels */,
    zero_point /* input zero point */,
    scale /* input scale */,
    output_zero_point /* output zero point */,
    output_scale /* output scale */,
    std::numeric_limits<uint8_t>::min() /* output min */,
    std::numeric_limits<uint8_t>::max() /* output max */,
    0 /* flags */,
    &tanh_op);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(tanh_op);

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK TanH operator");
  qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    output_scale,
    output_zero_point,
    input_contig.suggest_memory_format());

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_tanh_nc_q8(
    tanh_op,
    input_contig.size(0) /* batch size */,
    (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
    num_elems /* input stride */,
    (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
    num_elems /* output stride */);
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK TanH operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(tanh_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK TanH operator");
  return qy;
}
#endif  // USE_PYTORCH_QNNPACK

Tensor tanh_quantized_cpu(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    return qnnpack_tanh(qx);
  }
#endif  // USE_PYTORCH_QNNPACK
  Tensor qy;
  qtanh_stub(qx.device().type(), qx, qy);
  return qy;
}
}}  // namespace at::native

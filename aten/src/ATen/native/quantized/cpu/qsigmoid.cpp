#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>
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
#include <ATen/ops/sigmoid_native.h>
#endif

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qsigmoid_stub);

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_sigmoid(
    Tensor input, double output_scale, int64_t output_zero_point) {
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_sigmoid(): Got empty input tensor");
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_sigmoid(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));

  Tensor qy;
  initQNNPACK();

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }

  const auto zero_point = input_contig.q_zero_point();
  const auto scale = input_contig.q_scale();

  pytorch_qnnp_operator_t sigmoid_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_sigmoid_nc_q8(
    num_elems /* channels */,
    zero_point /* input zero point */,
    scale /* input scale */,
    output_zero_point /* output zero point */,
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    output_scale /* output scale */,
    std::numeric_limits<uint8_t>::min() /* output min */,
    std::numeric_limits<uint8_t>::max() /* output max */,
    0 /* flags */,
    &sigmoid_op);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(sigmoid_op);

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK sigmoid operator");
  qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    output_scale,
    output_zero_point,
    input_contig.suggest_memory_format());

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

// This ALWAYS outputs scale=1.0/256, dtype=quint8
// The zero_point is 0 for qint32 and quint8, but -128 for qint8.
Tensor sigmoid_quantized_cpu(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    constexpr double output_scale = 1.0f / 256.0f;
    constexpr int64_t output_zero_point = 0;
    return qnnpack_sigmoid(qx, output_scale, output_zero_point);
  }
#endif  // USE_PYTORCH_QNNPACK
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    // Naive implemenentation: uses dequantize/execute/quantize routine
    // - Output scale is set to 1.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    // See https://stackoverflow.com/a/34448562/3606192 for potential
    // optimizations
    double output_scale = 0.00390625;  // 1.0 / 2^8
    int64_t output_zero_point = 0;
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
    } else if (SCALAR_TYPE == at::kQInt8) {
      output_zero_point = -128;
    }
    qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
  });
  return qy;
}

namespace {

class QSigmoid final {
 public:
  static Tensor run(Tensor qx, double output_scale, int64_t output_zero_point) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    return qnnpack_sigmoid(qx, output_scale, output_zero_point);
  }
#endif  // USE_PYTORCH_QNNPACK
  Tensor qy;
  qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
  return qy;
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::sigmoid"), TORCH_FN(QSigmoid::run));
}
} // namespace

}}  // namespace at::native

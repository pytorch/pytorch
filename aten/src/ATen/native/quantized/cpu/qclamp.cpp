#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/quantized/Quantizer.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qclamp_stub);
DEFINE_DISPATCH(qclamp_min_stub);
DEFINE_DISPATCH(qclamp_max_stub);

namespace {

#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_clamp(Tensor input, const Scalar& min, const Scalar& max) {

  TORCH_CHECK(input.ndimension() > 0, "qnnpack_clamp(): Got empty input tensor");

  initQNNPACK();

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  size_t num_elems = 1;
  for (int i = 1; i < input_contig.ndimension(); ++i) {
    num_elems *= input_contig.size(i);
  }

  auto min_f = min.to<float>();
  auto max_f = max.to<float>();
  uint8_t min_q =
      at::native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), min_f).val_;
  uint8_t max_q =
      at::native::quantize_val<quint8>(input.q_scale(), input.q_zero_point(), max_f).val_;

  pytorch_qnnp_operator_t clamp_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_clamp_nc_u8(
    num_elems, // channels
    min_q,
    max_q,
    0, // flags
    &clamp_op);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(clamp_op);

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK Clamp operator");

  Tensor qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    input_contig.options(),
    input_contig.q_scale(),
    input_contig.q_zero_point());

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_clamp_nc_u8(
    clamp_op,
    input_contig.size(0), // batch_size
    (uint8_t*)input_contig.data_ptr<c10::quint8>(), // input_data
    num_elems, // input_stride
    (uint8_t*)qy.data_ptr<c10::quint8>(), // output_data
    num_elems); // output_stride
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK Clamp operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(clamp_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK Clamp operator");
  return qy;
}

#endif // USE_PYTORCH_QNNPACK

Tensor quantized_clamp_impl(
    const Tensor& qx,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  Tensor qy;
  if (min && max) {
#ifdef USE_PYTORCH_QNNPACK
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
        qx.scalar_type() == kQUInt8) {
      return qnnpack_clamp(qx, *min, *max);
    }
#endif
    qclamp_stub(qx.device().type(), qx, *min, *max, qy);
  } else {
#ifdef USE_PYTORCH_QNNPACK
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK) {
      TORCH_CHECK(
          false, "Both min and max should be specified for quantized clamp!");
    }
#endif
    if (max) {
      qclamp_max_stub(qx.device().type(), qx, *max, qy);
    } else if (min) {
      qclamp_min_stub(qx.device().type(), qx, *min, qy);
    } else {
      TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
    }
  }
  return qy;
}
} // namespace

// at::native functions for the native_functions.yaml
Tensor clamp_quantized_cpu(
    const Tensor& qx,
    const optional<Scalar>& min,
    const optional<Scalar>& max) {
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "clamp", [&]() {
    qy = quantized_clamp_impl(qx, min, max);
  });
  return qy;
}

// hardtanh is clamp with default min==-1.0f and default max==1.0f
Tensor hardtanh_quantized_cpu(
    const Tensor& qx,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl(qx, min, max);
  return qy;
}

Tensor& hardtanh_out_quantized_cpu(const Tensor& qx,
    const Scalar& min,
    const Scalar& max,
    Tensor& result) {
  result = quantized_clamp_impl(qx, min, max);
  return result;
}

Tensor& hardtanh_quantized_cpu_(
    Tensor& self,
    const Scalar& min,
    const Scalar& max) {
  Tensor qy;
  qy = quantized_clamp_impl(self, min, max);
  // This can be optimized in a future PR if it becomes a bottleneck.
  self.copy_(qy);
  return self;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::clamp"), TORCH_FN(clamp_quantized_cpu));
}

} // namespace native
} // namespace at

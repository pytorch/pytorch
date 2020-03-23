#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qadd_relu_stub);
DEFINE_DISPATCH(qadd_stub);
DEFINE_DISPATCH(qadd_scalar_relu_stub);
DEFINE_DISPATCH(qadd_scalar_stub);

namespace {

inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine,
      "Only per tensor quantization is suported in Add.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Add must have the same quantization shceme.");
  TORCH_CHECK(qa.numel() == qb.numel(), "Add operands must be the same size!");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "Add operands should have same data type.");
}

// Note: out is assumed to be the same size as self and other.
// Note: Addition is only supported when self, other, out are of the same dtype.
template <bool ReLUFused = false>
Tensor _add_out(Tensor& out, const Tensor& self, const Tensor& other) {
  if (ReLUFused) {
    qadd_relu_stub(self.device().type(), out, self, other);
  } else {
    qadd_stub(self.device().type(), out, self, other);
  }
  return out;
}

template <bool ReLUFused = false>
Tensor _add_scalar_out(Tensor& out, const Tensor& self, Scalar other) {
  TORCH_CHECK(
      self.qscheme() == kPerTensorAffine,
      "Only per tensor affine is supported for now!!");
  // To implement tensor-scalar addition in quantized space, we simply
  // adjust the quantization parameters based on the following rules:
  //
  // Let s = scale, z = zero point, c = other.toFloat(), c_q = round(c/s)
  // q_min = lowest representable value of scalar type
  // q_max = highest representable value of scalar type
  //
  // Let s' = the calculated scale or the output
  // z' = the calculated zero-point for the output
  //
  // If q_min > z - c_q
  //   s' = [(q_max - (z - c_q)]/[q_max - q_min] * s
  //   z' = q_min
  //   Xq' = at::requantize_from_int(Xq - z + c_q, s/s', z')
  // If q_max < z - c_q
  //   s' = [z - c_q -q_min]/[q_max - q_min] * s
  //   z' = q_max
  //   Xq' = at::requantize_from_int(Xq - z + c_q, s/s', z')
  // Else
  //   s' = s
  //   z' = z - c_q

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
    double s = self.q_scale();
    int64_t z = self.q_zero_point();
    double c = other.toDouble();
    int64_t q_min = std::numeric_limits<underlying_t>::min();
    int64_t q_max = std::numeric_limits<underlying_t>::max();

    int64_t c_q = std::nearbyint(c / s);

    double s_prime;
    int64_t z_prime;

    if (q_min > z - c_q) {
      s_prime = (((double)q_max - (z - c_q))) / ((double)q_max - q_min) * s;
      z_prime = q_min;
      out.set_quantizer_(make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        qadd_scalar_relu_stub(self.device().type(), out, self, c_q);
      } else {
        qadd_scalar_stub(self.device().type(), out, self, c_q);
      }
    } else if (q_max < z - c_q) {
      s_prime = ((double)(z - c_q) - q_min) / ((double)q_max - q_min) * s;
      z_prime = q_max;
      out.set_quantizer_(make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        qadd_scalar_relu_stub(self.device().type(), out, self, c_q);
      } else {
        qadd_scalar_stub(self.device().type(), out, self, c_q);
      }
    } else {
      s_prime = s;
      z_prime = z - c_q;
      out.copy_(self);
      out.set_quantizer_(make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        at::native::quantized_relu_(out);
      }
    }
  });
  return out;
}


template <bool ReLUFused = false>
class QAdd final : public c10::OperatorKernel {
#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_add(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  TORCH_CHECK(qa.ndimension() > 0, "qnnpack_add(): Got empty input tensor.");
  Tensor qa_contig = qa.contiguous();
  Tensor qb_contig = qb.contiguous();

  const auto a_zero_point = qa_contig.q_zero_point();
  const auto b_zero_point = qb_contig.q_zero_point();
  const auto a_scale = qa_contig.q_scale();
  const auto b_scale = qb_contig.q_scale();

  Tensor qy = at::_empty_affine_quantized(
      qa_contig.sizes(), at::device(kCPU).dtype(kQUInt8), scale, zero_point);

  if (qa_contig.size(0) == 0) {
    return qy;
  }

  initQNNPACK();

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  size_t num_elems = qa_contig.numel() / qa_contig.size(0);
  auto output_min = ReLUFused
      ? activationLimits(scale, zero_point, Activation::RELU)
            .first
      : std::numeric_limits<uint8_t>::min();
  auto output_max = ReLUFused
      ? activationLimits(scale, zero_point, Activation::RELU)
            .second
      : std::numeric_limits<uint8_t>::max();
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_add_nc_q8(
      num_elems /* input size */,
      a_zero_point /* a zero_point */,
      a_scale /* a scale */,
      b_zero_point /* b zero_point */,
      b_scale /* b scale */,
      static_cast<uint8_t>(zero_point) /* sum zero_point */,
      scale /* sum scale */,
      output_min /* output min */,
      output_max /* output max */,
      0 /* flags */,
      &qnnpack_operator);

  TORCH_INTERNAL_ASSERT(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK Add operator");

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_add_nc_q8(
      qnnpack_operator /* add op */,
      qa_contig.size(0) /* batch size */,
      (uint8_t*)qa_contig.data_ptr<c10::quint8>() /* a data */,
      num_elems /* A stride */,
      (uint8_t*)qb_contig.data_ptr<c10::quint8>() /* b data */,
      num_elems /* B stride */,
      (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
      num_elems /* sum stride */);
  TORCH_INTERNAL_ASSERT(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Add operator");

  pthreadpool_t threadpool = caffe2::mobile_pthreadpool();
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Add operator");

  return qy;
}
#endif
 public:
  Tensor operator()(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
    check_inputs(qa, qb);
#ifdef USE_PYTORCH_QNNPACK
    if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
        qa.scalar_type() == kQUInt8 && qb.scalar_type() == kQUInt8) {
      return qnnpack_add(qa, qb, scale, zero_point);
    }
#endif
    auto qc = at::_empty_affine_quantized(
        qa.sizes(),
        at::device(kCPU).dtype(qa.scalar_type()),
        scale,
        zero_point,
        qa.suggest_memory_format());
    return _add_out<ReLUFused>(qc, qa, qb);
  }
};

template <bool ReLUFused = false>
class QAddOut final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Tensor qb, Tensor out) {
    check_inputs(qa, qb);
    check_inputs(qa, out);
    return _add_out<ReLUFused>(out, qa, qb);
  }
};


template <bool ReLUFused = false>
class QAddScalar final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Scalar b) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is suuported in Add.");
    auto qc = at::empty_like(qa, qa.suggest_memory_format());
    return _add_scalar_out<ReLUFused>(qc, qa, b);
  }
};

template <bool ReLUFused = false>
class QAddScalarOut final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Scalar b, Tensor out) {
    check_inputs(qa, out);
    return _add_scalar_out<ReLUFused>(out, qa, b);
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::add(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAdd</*ReLUFused=*/false>>(DispatchKey::QuantizedCPUTensorId))
.op("quantized::add_relu(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAdd</*ReLUFused=*/true>>(DispatchKey::QuantizedCPUTensorId))
.op("quantized::add_out(Tensor qa, Tensor qb, Tensor(a!) out)"
     "-> Tensor(a!) out",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAddOut</*ReLUFused=*/false>>(DispatchKey::QuantizedCPUTensorId))
.op("quantized::add_relu_out(Tensor qa, Tensor qb, Tensor(a!) out)"
     "-> Tensor(a!) out",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAddOut</*ReLUFused=*/true>>(DispatchKey::QuantizedCPUTensorId))
.op("quantized::add_scalar(Tensor qa, Scalar b) -> Tensor qc",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAddScalar</*ReLUFused=*/false>>(DispatchKey::QuantizedCPUTensorId))
.op("quantized::add_scalar_relu(Tensor qa, Scalar b) -> Tensor qc",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAddScalar</*ReLUFused=*/true>>(DispatchKey::QuantizedCPUTensorId))
.op("quantized::add_scalar_out(Tensor qa, Scalar b, Tensor(a!) out)"
     "-> Tensor(a!) out",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAddScalarOut</*ReLUFused=*/false>>(DispatchKey::QuantizedCPUTensorId))
.op("quantized::add_scalar_relu_out(Tensor qa, Scalar b, Tensor(a!) out)"
     "-> Tensor(a!) out",
    c10::RegisterOperators::options()
      .aliasAnalysis(at::AliasAnalysisKind::FROM_SCHEMA)
      .kernel<QAddScalarOut</*ReLUFused=*/true>>(DispatchKey::QuantizedCPUTensorId));
}  // namespace
}}  // namespace at::native

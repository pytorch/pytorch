#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/_empty_affine_quantized_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/relu_native.h>
#endif

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
Tensor _add_scalar_out(Tensor& out, const Tensor& self, const Scalar& other) {
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
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    int64_t q_min = std::numeric_limits<underlying_t>::min();
    int64_t q_max = std::numeric_limits<underlying_t>::max();

    int64_t c_q = std::nearbyint(c / s);

    double s_prime;
    int64_t z_prime;

    if (q_min > z - c_q) {
      s_prime = (((double)q_max - (z - c_q))) / ((double)q_max - q_min) * s;
      z_prime = q_min;
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        qadd_scalar_relu_stub(self.device().type(), out, self, c_q);
      } else {
        qadd_scalar_stub(self.device().type(), out, self, c_q);
      }
    } else if (q_max < z - c_q) {
      s_prime = ((double)(z - c_q) - q_min) / ((double)q_max - q_min) * s;
      z_prime = q_max;
      set_quantizer_(out, make_per_tensor_affine_quantizer(
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
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          s_prime, z_prime, self.scalar_type()));
      if (ReLUFused) {
        at::native::relu_quantized_cpu_(out);
      }
    }
  });
  return out;
}


#ifdef USE_PYTORCH_QNNPACK
template <bool ReLUFused = false>
Tensor qnnpack_add(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  TORCH_CHECK(qa.ndimension() > 0, "qnnpack_add(): Got empty input tensor.");
  TORCH_CHECK(qa.scalar_type() == c10::kQUInt8 && qb.scalar_type() == c10::kQUInt8,
                "qnnpack_add(): Expected both input data types to be ",
                toString(c10::kQUInt8),
                " but got ",
                toString(qa.scalar_type()),
                " and ",
                toString(qb.scalar_type()));
  Tensor qa_contig = qa.contiguous(qa.suggest_memory_format());
  // Reason for use qa's memory format for qb is that for the underlying
  // kernel can flatten all the dims and iterate over both the tensors.
  // In most cases, both qa and qb are in same memory format.
  // When they are not there is a copy overhead to make it contiguous
  // in qa's memory format.
  Tensor qb_contig = qb.contiguous(qa.suggest_memory_format());

  const auto a_zero_point = qa_contig.q_zero_point();
  const auto b_zero_point = qb_contig.q_zero_point();
  const auto a_scale = qa_contig.q_scale();
  const auto b_scale = qb_contig.q_scale();

  Tensor qy = at::native::empty_affine_quantized(
      qa_contig.sizes(),
      kQUInt8,
      c10::nullopt /* layout */,
      kCPU,
      c10::nullopt /* pin_memory */,
      scale,
      zero_point,
      qa.suggest_memory_format());

  if (qa_contig.size(0) == 0) {
    return qy;
  }

  initQNNPACK();

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};

  size_t num_elems = qa_contig.numel() / qa_contig.size(0);
  auto output_min = ReLUFused
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      ? activationLimits<uint8_t>(scale, zero_point, Activation::RELU)
            .first
      : std::numeric_limits<uint8_t>::min();
  auto output_max = ReLUFused
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      ? activationLimits<uint8_t>(scale, zero_point, Activation::RELU)
            .second
      : std::numeric_limits<uint8_t>::max();
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_add_nc_q8(
      num_elems /* input size */,
      a_zero_point /* a zero_point */,
      a_scale /* a scale */,
      b_zero_point /* b zero_point */,
      b_scale /* b scale */,
      static_cast<uint8_t>(zero_point) /* sum zero_point */,
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
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

  pthreadpool_t threadpool = caffe2::pthreadpool_();
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);

  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Add operator");

  return qy;
}
#endif

template <bool ReLUFused = false>
Tensor qadd(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
  check_inputs(qa, qb);
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qa.sizes() == qb.sizes() && /* qnnpack does not support boradcasting */
      qa.scalar_type() == kQUInt8 && qb.scalar_type() == kQUInt8) {
    return qnnpack_add<ReLUFused>(qa, qb, scale, zero_point);
  }
#endif
  auto qc = at::_empty_affine_quantized(
      qa.sizes(),
      at::device(kCPU)
         .dtype(qa.scalar_type())
         .memory_format(qa.suggest_memory_format()),
      scale,
      zero_point,
      c10::nullopt);
  return _add_out<ReLUFused>(qc, qa, qb);
}

template <bool ReLUFused = false>
Tensor qadd_out(Tensor qa, Tensor qb, Tensor out) {
  check_inputs(qa, qb);
  check_inputs(qa, out);
  return _add_out<ReLUFused>(out, qa, qb);
}


template <bool ReLUFused = false>
Tensor qadd_scalar(Tensor qa, const Scalar& b) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Add.");
  auto qc = at::empty_like(qa, qa.suggest_memory_format());
  return _add_scalar_out<ReLUFused>(qc, qa, b);
}

template <bool ReLUFused = false>
Tensor qadd_scalar2(Scalar b, Tensor qa) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Add.");
  auto qc = at::empty_like(qa, qa.suggest_memory_format());
  return _add_scalar_out<ReLUFused>(qc, qa, b);
}

template <bool ReLUFused = false>
Tensor qadd_scalar_out(Tensor qa, const Scalar& b, Tensor out) {
  check_inputs(qa, out);
  return _add_scalar_out<ReLUFused>(out, qa, b);
}

// `torch.jit.trace` will trace Scalar as Tensor
// This can be removed after broadcast is supported and
// all variations of `quantized::add` is merged into `quantized::add`
template <bool ReLUFused = false>
Tensor qadd_scalar_tensor(Tensor qa, Tensor b) {
  return qadd_scalar(qa, b.item());
}

// `torch.jit.trace` will trace Scalar as Tensor
// This can be removed after broadcast is supported and
// all variations of `quantized::add` is merged into `quantized::add`
template <bool ReLUFused = false>
Tensor qadd_scalar_tensor_out(Tensor qa, Tensor b, Tensor out) {
  return qadd_scalar_out(qa, b.item(), out);
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::add"),                 TORCH_FN(qadd</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.out"),             TORCH_FN(qadd_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar"),          TORCH_FN(qadd_scalar</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar2"),          TORCH_FN(qadd_scalar2</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add.Scalar_out"),      TORCH_FN(qadd_scalar_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"),            TORCH_FN(qadd</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.out"),        TORCH_FN(qadd_out</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar"),     TORCH_FN(qadd_scalar</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar2"),     TORCH_FN(qadd_scalar2</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu.Scalar_out"), TORCH_FN(qadd_scalar_out</*ReLUFused=*/true>));
  // deprecated functions, kept for backward compatibility
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_out"),             TORCH_FN(qadd_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu_out"),        TORCH_FN(qadd_out</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar"),          TORCH_FN(qadd_scalar</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu"),     TORCH_FN(qadd_scalar</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_out"),      TORCH_FN(qadd_scalar_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu_out"), TORCH_FN(qadd_scalar_out</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar.Tensor"),   TORCH_FN(qadd_scalar_tensor</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu.Tensor"), TORCH_FN(qadd_scalar_tensor</*ReLUFused=*/true>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_out.Tensor"), TORCH_FN(qadd_scalar_tensor_out</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_scalar_relu_out.Tensor"), TORCH_FN(qadd_scalar_tensor_out</*ReLUFused=*/true>));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::add"), TORCH_FN(qadd</*ReLUFused=*/false>));
}

}  // namespace

Tensor quantized_add(Tensor qa, Tensor qb, double scale, int64_t zero_point){
  return qadd<false>(qa, qb, scale, zero_point);
}

}}  // namespace at::native

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
  if (ReLUFused) {
    qadd_scalar_relu_stub(self.device().type(), out, self, other);
  } else {
    qadd_scalar_stub(self.device().type(), out, self, other);
  }
  return out;
}


template <bool ReLUFused = false>
class QAdd final : public torch::OperatorKernel {
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
class QAddOut final : public torch::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Tensor qb, Tensor out) {
    check_inputs(qa, qb);
    check_inputs(qa, out);
    return _add_out<ReLUFused>(out, qa, qb);
  }
};


template <bool ReLUFused = false>
class QAddScalar final : public torch::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Scalar b) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is suuported in Add.");
    auto qc = at::empty_like(qa);
    return _add_scalar_out<ReLUFused>(qc, qa, b);
  }
};

template <bool ReLUFused = false>
class QAddScalarOut final : public torch::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Scalar b, Tensor out) {
    check_inputs(qa, out);
    return _add_scalar_out<ReLUFused>(out, qa, b);
  }
};

} // namespace

constexpr const char *kTensorTensorError = "torch.add is not supported when adding two"
" quantized tensors. Please use torch.nn.quantized.modules.FloatFunctional";
constexpr const char *kAlphaError = "The alpha parameter with values != 1.0 is currently"
" not supported with quantized tensors";

// ATen bindings for add

Tensor& quantized_add_out(Tensor& out, const Tensor& self, const Tensor& other, Scalar alpha) {
  TORCH_CHECK(other.sizes().size() == 0, kTensorTensorError);
  TORCH_CHECK(alpha.toFloat() == 1.0, kAlphaError);
  _add_scalar_out(out, self, other.item());
  return out;
}

Tensor quantized_add_scalar(const Tensor& self, Scalar other) {
  Tensor retval = at::empty_like(self);
  _add_scalar_out(retval, self, other);
  return retval;
}

Tensor& quantized_add_scalar_out(Tensor& out, const Tensor& self, Scalar other) {
  _add_scalar_out(out, self, other);
  return out;
}

Tensor& quantized_add_scalar_(Tensor& self, Scalar other) {
  _add_scalar_out(self, self, other);
  return self;
}

Tensor quantized_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  TORCH_CHECK(other.sizes().size() == 0, kTensorTensorError);
  TORCH_CHECK(alpha.toFloat() == 1.0, kAlphaError);
  Tensor retval = at::empty_like(self);
  _add_scalar_out(retval, self, other.item());
  return retval;
}

// ATen bindings for AddRelU

Tensor& quantized_add_relu_out(Tensor& out, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(other.sizes().size() == 0, kTensorTensorError);
  _add_scalar_out</*ReLUFused=*/true>(out, self, other.item());
  return out;
}

Tensor quantized_add_scalar_relu(const Tensor& self, Scalar other) {
  Tensor retval = at::empty_like(self);
  _add_scalar_out</*ReLUFused=*/true>(retval, self, other);
  return retval;
}

Tensor& quantized_add_scalar_relu_out(Tensor& out, const Tensor& self, Scalar other) {
  _add_scalar_out</*ReLUFused=*/true>(out, self, other);
  return out;
}

Tensor& quantized_add_scalar_relu_(Tensor& self, Scalar other) {
  _add_scalar_out</*ReLUFused=*/true>(self, self, other);
  return self;
}

Tensor quantized_add_relu(const Tensor& self, const Tensor& other, Scalar alpha) {
  TORCH_CHECK(other.sizes().size() == 0, kTensorTensorError);
  TORCH_CHECK(alpha.toFloat() == 1.0, kAlphaError);
  Tensor retval = at::empty_like(self);
  _add_scalar_out</*ReLUFused=*/true>(retval, self, other.item());
  return retval;
}
  
}}  // namespace at::native

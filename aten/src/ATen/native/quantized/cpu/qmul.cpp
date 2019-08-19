#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>

namespace at {
namespace native {
namespace {

inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is suuported in Mul.");
  TORCH_CHECK(qa.qscheme() == qb.qscheme(),
              "Both inputs to Mul must have the same quantization shceme.");
  TORCH_CHECK(qa.numel() == qb.numel(),
              "Mul operands must be the same size!");
  TORCH_CHECK(qa.scalar_type() == qb.scalar_type(),
              "Mul operands should have same data type.");
}

// Note: out is assumed to be the same size as self and other.
// Note: Multiplication is only supported when self, other, out are of the same
//       dtype.
template <bool ReLUFused = false>
Tensor _mul_out(Tensor& out, const Tensor& self, const Tensor& other) {
  int64_t zero_point = out.q_zero_point();
  double scale = out.q_scale();
  int64_t self_zero_point = self.q_zero_point();
  double self_scale = self.q_scale();
  int64_t other_zero_point = other.q_zero_point();
  double other_scale = other.q_scale();

  auto iter = TensorIterator::binary_op(out, self, other);
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul", [&]() {
    cpu_kernel(iter, [&](scalar_t a, scalar_t b) -> scalar_t {
      const auto da = at::dequantize_val(self_scale, self_zero_point, a);
      const auto db = at::dequantize_val(other_scale, other_zero_point, b);
      float c = da * db;
      if (ReLUFused) {
        c = std::max<float>(c, 0.0);
      }
      return at::quantize_val<scalar_t>(scale, zero_point, c);
    });
  });
  return out;
}

template <bool ReLUFused = false>
Tensor _mul_scalar_out(Tensor& out, const Tensor& self, Scalar other) {
  int64_t zero_point = out.q_zero_point();
  double scale = out.q_scale();
  int64_t self_zero_point = self.q_zero_point();
  double self_scale = self.q_scale();

  auto iter = TensorIterator::unary_op(out, self);
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul", [&]() {
    cpu_kernel(iter, [&](scalar_t a) -> scalar_t {
      const auto da = at::dequantize_val(self_scale, self_zero_point, a);
      float c = da * other.toFloat();
      if (ReLUFused) {
        c = std::max<float>(c, 0.0);
      }
      return at::quantize_val<scalar_t>(scale, zero_point, c);
    });
  });
  return out;
}

template <bool ReLUFused = false>
class QMul final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Tensor qb,
                    double scale, int64_t zero_point) {
    check_inputs(qa, qb);
    auto qc = at::_empty_affine_quantized(qa.sizes(),
      at::device(kCPU).dtype(qa.scalar_type()), scale, zero_point);
    return _mul_out<ReLUFused>(qc, qa, qb);
  }
};

template <bool ReLUFused = false>
class QMulOut final : public c10::OperatorKernel {
 public:
  Tensor operator()(at::Tensor qa, at::Tensor qb, Tensor out) {
    check_inputs(qa, qb);
    return _mul_out<ReLUFused>(out, qa, qb);
  }
};


template <bool ReLUFused = false>
class QMulScalar final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Scalar b,
                    double scale, int64_t zero_point) {
    TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is suuported in Mul.");
    auto qc = at::_empty_affine_quantized(qa.sizes(),
      at::device(kCPU).dtype(qa.scalar_type()), scale, zero_point);
    return _mul_scalar_out<ReLUFused>(qc, qa, b);
  }
};

template <bool ReLUFused = false>
class QMulScalarOut final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Scalar b, Tensor out) {
    check_inputs(qa, out);
    return _mul_scalar_out<ReLUFused>(out, qa, b);
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::mul(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QMul</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::mul_relu(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QMul</*ReLUFused=*/true>>(QuantizedCPUTensorId()))
.op("quantized::mul_out(Tensor qa, Tensor qb, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QMulOut</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::mul_relu_out(Tensor qa, Tensor qb, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QMulOut</*ReLUFused=*/true>>(QuantizedCPUTensorId()))

.op("quantized::mul_scalar(Tensor qa, Scalar b, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QMulScalar</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::mul_scalar_relu(Tensor qa, Scalar b, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QMulScalar</*ReLUFused=*/true>>(QuantizedCPUTensorId()))
.op("quantized::mul_scalar_out(Tensor qa, Scalar b, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QMulScalarOut</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::mul_scalar_relu_out(Tensor qa, Scalar b, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QMulScalarOut</*ReLUFused=*/true>>(QuantizedCPUTensorId()))
;
}  // namespace
}}  // namespace at::native

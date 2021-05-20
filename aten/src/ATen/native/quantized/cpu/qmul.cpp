#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(qmul_relu_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(qmul_stub);

namespace {

inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine,
              "Only per tensor quantization is supported in Mul.");
  TORCH_CHECK(qa.scalar_type() == qb.scalar_type(),
              "Mul operands should have same data type.");
  TORCH_CHECK(qa.qscheme() == qb.qscheme(),
              "Both inputs to Mul must have the same quantization shceme.");
}

// Note: out is assumed to be the same size as self and other.
// Note: Multiplication is only supported when self, other, out are of the same
//       dtype.
template <bool ReLUFused = false>
Tensor _mul_out(Tensor& out, const Tensor& self, const Tensor& other) {
  if (ReLUFused) {
    qmul_relu_stub(self.device().type(), out, self, other);
  } else {
    qmul_stub(self.device().type(), out, self, other);
  }
  return out;
}

template <bool ReLUFused = false>
Tensor _mul_scalar_out(Tensor& out, const Tensor& self, const Scalar& other) {
  int64_t self_zero_point = self.q_zero_point();
  double self_scale = self.q_scale();
  double other_val = other.toDouble();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  double scale_prime;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t zero_point_prime;

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qmul_scalar", [&]() {
    // NOLINTNEXTLINE(bugprone-signed-char-misuse)
    int64_t q_min = std::numeric_limits<underlying_t>::min();
    int64_t q_max = std::numeric_limits<underlying_t>::max();

    if (other_val > 0.0) {
      scale_prime = other_val * self_scale;
      zero_point_prime = self_zero_point;

      if (ReLUFused) {
        qrelu_stub(self.device().type(), self, out);
      } else {
        out.copy_(self);
      }
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          scale_prime, zero_point_prime, self.scalar_type()));
    } else if (other_val == 0.0) {
      scale_prime = 1.0;
      zero_point_prime = 0;

      // Strided "memset"
      // Set all values to 0
      auto iter = TensorIterator::unary_op(out, self);
      cpu_kernel_vec(
          iter,
          [&](scalar_t a) -> scalar_t { return scalar_t(0); },
          [&](Vec256<scalar_t> vec) -> Vec256<scalar_t> {
            return Vec256<scalar_t>(scalar_t(0));
          });
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          scale_prime, zero_point_prime, self.scalar_type()));
    } else /* other_val < 0.0 */ {
      scale_prime = std::abs(other_val) * self_scale;
      zero_point_prime = q_max - (self_zero_point - q_min);

      // xq' = q_max + q_min - x_q
      auto iter = TensorIterator::unary_op(out, self);
      cpu_kernel(
          iter,
          [&](scalar_t a) -> scalar_t {
            a = scalar_t(underlying_t(q_max + q_min - a.val_));
            if (ReLUFused) {
              a = scalar_t(std::max(a.val_, underlying_t(zero_point_prime)));
            }
            return a;
          });
      set_quantizer_(out, make_per_tensor_affine_quantizer(
          scale_prime, zero_point_prime, self.scalar_type()));
    }
  });

  return out;
}

template <bool ReLUFused = false>
class QMul final {
 public:
  static Tensor run(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
    check_inputs(qa, qb);
    auto qc = at::_empty_affine_quantized(
        infer_size_dimvector(qa.sizes(), qb.sizes()),
        at::device(kCPU).dtype(qa.scalar_type()),
        scale,
        zero_point,
        qa.suggest_memory_format());
    return _mul_out<ReLUFused>(qc, qa, qb);
  }
};

template <bool ReLUFused = false>
class QMulOut final {
 public:
  static Tensor run(at::Tensor qa, at::Tensor qb, Tensor out) {
    check_inputs(qa, qb);
    return _mul_out<ReLUFused>(out, qa, qb);
  }
};


template <bool ReLUFused = false>
class QMulScalar final {
 public:
  static Tensor run(Tensor qa, const Scalar& b) {
    TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Mul.");
    auto qc = at::empty_like(qa, qa.suggest_memory_format());
    return _mul_scalar_out<ReLUFused>(qc, qa, b);
  }
};

template <bool ReLUFused = false>
class QMulScalar2 final {
 public:
  static Tensor run(const Scalar& b, Tensor qa) {
    TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is supported in Mul.");
    auto qc = at::empty_like(qa, qa.suggest_memory_format());
    return _mul_scalar_out<ReLUFused>(qc, qa, b);
  }
};

template <bool ReLUFused = false>
class QMulScalarOut final {
 public:
  static Tensor run(Tensor qa, const Scalar& b, Tensor out) {
    check_inputs(qa, out);
    return _mul_scalar_out<ReLUFused>(out, qa, b);
  }
};

// `torch.jit.trace` will trace Scalar as Tensor
// This can be removed after broadcast is supported and
// all variations of `quantized::mul` is merged into `quantized::mul`
template <bool ReLUFused = false>
class QMulScalarTensor final {
 public:
  static Tensor run(Tensor qa, Tensor b) {
    TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is suported in Mul.");
    auto qc = at::empty_like(qa, qa.suggest_memory_format());
    return _mul_scalar_out<ReLUFused>(qc, qa, b.item());
  }
};

// `torch.jit.trace` will trace Scalar as Tensor
// This can be removed after broadcast is supported and
// all variations of `quantized::mul` is merged into `quantized::mul`
template <bool ReLUFused = false>
class QMulScalarTensorOut final {
 public:
  static Tensor run(Tensor qa, Tensor b, Tensor out) {
    check_inputs(qa, out);
    return _mul_scalar_out<ReLUFused>(out, qa, b.item());
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul"),                 TORCH_FN(QMul</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.out"),             TORCH_FN(QMulOut</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar"),          TORCH_FN(QMulScalar</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar2"),          TORCH_FN(QMulScalar2</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul.Scalar_out"),      TORCH_FN(QMulScalarOut</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu"),            TORCH_FN(QMul</*ReLUFused=*/true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.out"),        TORCH_FN(QMulOut</*ReLUFused=*/true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar"),     TORCH_FN(QMulScalar</*ReLUFused=*/true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar2"),     TORCH_FN(QMulScalar2</*ReLUFused=*/true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu.Scalar_out"), TORCH_FN(QMulScalarOut</*ReLUFused=*/true>::run));
  // deprecated functions, kept for backward compatibility
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_out"),             TORCH_FN(QMulOut</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_relu_out"),        TORCH_FN(QMulOut</*ReLUFused=*/true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar"),          TORCH_FN(QMulScalar</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu"),     TORCH_FN(QMulScalar</*ReLUFused=*/true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_out"),      TORCH_FN(QMulScalarOut</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu_out"), TORCH_FN(QMulScalarOut</*ReLUFused=*/true>::run));
  // TODO: remove after broadcasting is supported
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar.Tensor"), TORCH_FN(QMulScalarTensor</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu.Tensor"), TORCH_FN(QMulScalarTensor</*ReLUFused=*/true>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_out.Tensor"), TORCH_FN(QMulScalarTensorOut</*ReLUFused=*/false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::mul_scalar_relu_out.Tensor"), TORCH_FN(QMulScalarTensorOut</*ReLUFused=*/true>::run));
}

}  // namespace
}}  // namespace at::native

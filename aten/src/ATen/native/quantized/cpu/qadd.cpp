#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qadd_relu_stub);
DEFINE_DISPATCH(qadd_stub);

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
  TORCH_CHECK(self.qscheme() == kPerTensorAffine,
              "Only per tensor affine is supported for now!!");
  // To implement tensor-scalar addition in quantized space, we simply
  // adjust the quantization parameters based on the following rules:
  //
  // Let s = scale, z = zero point, c = other.toFloat(),
  //     q_min = lowest representable value of scalar type
  //     q_max = highest representable value of scalar type
  //
  //  Let s' = the calculated scale or the output
  //      z' = the calculated zero-point for the output
  //
  //  If q_min > z - round(c / s)
  //    s' = [(q_max - z)*s + c]/[q_max - q_min]
  //    z' = q_min
  //
  //  If q_max < z - round(c / s)
  //    s' = -[(q_min - z)*s + c]/[q_max - q_min]
  //    z' = q_max
  //
  //  Else
  //    s' = s
  //    z = z - round(c / s)

  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd_scalar", [&]() {
    double s = self.q_scale();
    int64_t z = self.q_zero_point();
    float c = other.toFloat();
    int64_t q_min = std::numeric_limits<underlying_t>::min();
    int64_t q_max = std::numeric_limits<underlying_t>::max();

    int64_t rounded = std::nearbyint(c / s);
    if (rounded > q_max || rounded < q_min) {
      throw std::runtime_error("Out of range");
    }
    int64_t shifted_z = z - rounded;

    double s_prime;
    int64_t z_prime;

    if (q_min > shifted_z) {
      s_prime = ((q_max - z)*s + c)/(q_max - q_min);
      z_prime = q_min;
    } else if (q_max < shifted_z) {
      s_prime = -((q_min - z)*s + c)/(q_max - q_min);
      z_prime = q_max;
    } else {
      s_prime = s;
      z_prime = shifted_z;
    }

    out.copy_(self);
    out.set_quantizer_(make_per_tensor_affine_quantizer(s_prime, z_prime, self.scalar_type()));
  });
  return out;
}


template <bool ReLUFused = false>
class QAdd final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Tensor qb, double scale, int64_t zero_point) {
    check_inputs(qa, qb);
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
    auto qc = at::_empty_affine_quantized(qa.sizes(),
      at::device(kCPU).dtype(
        qa.scalar_type()),
        1.0,
        0,
        qa.suggest_memory_format());
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
      .kernel<QAdd</*ReLUFused=*/false>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_relu(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAdd</*ReLUFused=*/true>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_out(Tensor qa, Tensor qb, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddOut</*ReLUFused=*/false>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_relu_out(Tensor qa, Tensor qb, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddOut</*ReLUFused=*/true>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_scalar(Tensor qa, Scalar b)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAddScalar</*ReLUFused=*/false>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_scalar_relu(Tensor qa, Scalar b) -> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAddScalar</*ReLUFused=*/true>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_scalar_out(Tensor qa, Scalar b, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddScalarOut</*ReLUFused=*/false>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_scalar_relu_out(Tensor qa, Scalar b, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddScalarOut</*ReLUFused=*/true>>(TensorTypeId::QuantizedCPUTensorId));
}  // namespace
}}  // namespace at::native

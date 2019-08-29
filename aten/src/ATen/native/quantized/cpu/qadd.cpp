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
  int64_t zero_point = out.q_zero_point();
  double scale = out.q_scale();
  int64_t self_zero_point = self.q_zero_point();
  double self_scale = self.q_scale();
  int64_t other_zero_point = other.q_zero_point();
  double other_scale = other.q_scale();

  auto iter = TensorIterator::binary_op(out, self, other);
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
    cpu_kernel(iter, [&](scalar_t a, scalar_t b) -> scalar_t {
      const auto da = at::dequantize_val(self_scale, self_zero_point, a);
      const auto db = at::dequantize_val(other_scale, other_zero_point, b);
      float c = da + db;
      if (ReLUFused) {
        c = std::max<float>(c, 0.0);
      }
      return at::quantize_val<scalar_t>(scale, zero_point, c);
    });
  });
  return out;
}

template <bool ReLUFused = false>
Tensor _add_scalar_(Tensor& self, Scalar other) {
  TORCH_CHECK(self.qscheme() == kPerTensorAffine,
              "Only per tensor affine is supported for now!!");
  float other_val = other.toFloat();
  if (other_val <= std::numeric_limits<float>::epsilon()) {
    return self;
  }
  double s = self.q_scale();
  int64_t z = self.q_zero_point();
  int64_t qmin, qmax;
  double xmin, xmax;
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "qadd", [&]() {
    qmin = std::numeric_limits<underlying_t>::min();
    qmax = std::numeric_limits<underlying_t>::max();
    xmin = at::dequantize_val(self.q_scale(), self.q_zero_point(),
                              static_cast<scalar_t>(qmin));
    xmax = at::dequantize_val(self.q_scale(), self.q_zero_point(),
                              static_cast<scalar_t>(qmax));
  });

  int64_t new_z = z;
  double new_s = s;

  if (other_val > 0) {  // Addition
    if (other_val < (z - qmin) * s) {
      new_z = new_z - std::nearbyint(other_val / s);
    } else {
      new_z = qmin;
      new_s = (xmax + other_val) / (qmax - qmin);
    }
  } else {  // Subtraction
    if (-other_val < (qmax - z) * s) {
      new_z = new_z - std::nearbyint(other_val / s);
    } else {
      new_z = qmax;
      new_s = -(xmin + other_val) / (qmax - qmin);
    }
  }
  // double xmin = (double(qmin) - self.q_zero_point()) * self.q_scale();
  // double new_s = s * ((std::max<int64_t>(qmax - z, 0) - std::min<int64_t>(qmin - z, 0))
  //                / (qmax - qmin));
  // int64_t new_z = qmin - std::min<int64_t>(xmin + other_int, 0) / new_s;
  // new_z = std::min(qmax, new_z);
  // new_z = std::max(qmin, new_z);
  self.set_quantizer_(make_per_tensor_affine_quantizer(new_s, new_z,
                                                       self.scalar_type()));

  if (ReLUFused) {
    return at::relu_(self);
  }
  return self;
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
        zero_point);
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
  Tensor operator()(Tensor qa, Scalar b,
                    double scale, int64_t zero_point) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine,
              "Only per tensor quantization is suuported in Add.");
    auto out = qa.clone();
    return _add_scalar_<ReLUFused>(out, b);
  }
};

template <bool ReLUFused = false>
class QAddScalarOut final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Scalar b, Tensor out) {
    check_inputs(qa, out);
    out = qa.clone();
    return _add_scalar_<ReLUFused>(out, b);
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::add(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAdd</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::add_relu(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAdd</*ReLUFused=*/true>>(QuantizedCPUTensorId()))
.op("quantized::add_out(Tensor qa, Tensor qb, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddOut</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::add_relu_out(Tensor qa, Tensor qb, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddOut</*ReLUFused=*/true>>(QuantizedCPUTensorId()))
.op("quantized::add_scalar(Tensor qa, Scalar b, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAddScalar</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::add_scalar_relu(Tensor qa, Scalar b, float scale,"
     "int zero_point) -> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAddScalar</*ReLUFused=*/true>>(QuantizedCPUTensorId()))
.op("quantized::add_scalar_out(Tensor qa, Scalar b, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddScalarOut</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::add_scalar_relu_out(Tensor qa, Scalar b, Tensor out)"
     "-> Tensor out",
    c10::RegisterOperators::options()
      .kernel<QAddScalarOut</*ReLUFused=*/true>>(QuantizedCPUTensorId()));
}  // namespace
}}  // namespace at::native

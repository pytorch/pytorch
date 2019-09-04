#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpu/vec256/vec256.h>
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

  // Broadcast out the parameters here to amortize out that cost across
  // loop iterations.
  // TODO: we can optimize dequantization by doing a premultiplication
  // of the zero point by scale and doing FMA on scale*x_q - (scale*zero_point)
  auto self_zero_point_vec = Vec256<float>((float)self_zero_point);
  auto self_scale_vec = Vec256<float>(self_scale);
  auto other_zero_point_vec = Vec256<float>((float)other_zero_point);
  auto other_scale_vec = Vec256<float>(other_scale);

  auto iter = TensorIterator::binary_op(out, self, other);

  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
    using Vec = Vec256<scalar_t>;
    cpu_kernel_vec(iter, [&](scalar_t a, scalar_t b) -> scalar_t {
      const auto da = at::dequantize_val(self_scale, self_zero_point, a);
      const auto db = at::dequantize_val(other_scale, other_zero_point, b);
      float c = da + db;
      if (ReLUFused) {
        c = std::max<float>(c, 0.0);
      }
      return at::quantize_val<scalar_t>(scale, zero_point, c);
    },
    [&](Vec a, Vec b) -> Vec {
      const auto da = a.dequantize(self_scale_vec, self_zero_point_vec);
      const auto db = b.dequantize(other_scale_vec, other_zero_point_vec);
      Vec::float_vec_return_type retvals;
      for (int i = 0; i < Vec::float_num_vecs(); ++i) {
        auto c = da[i] + db[i];
        if (ReLUFused) {
          c = vec256::maximum(c, Vec256<float>(0.0f));
        }
        retvals[i] = c;
      }
      // TODO: fbgemm::Quantize doesn't support taking in the pre-broadcasted
      // parameters. We might be able to save some cycles by enabling that
      // in the API.
      // TODO: specialize fbgemm::Quantize for a single vector and make it
      // inlineable. This could help with interleaving as suggested by the
      // TensorIterator implementations
      auto rv = Vec::quantize(retvals, scale, zero_point);
      return rv;
    });
  });
  return out;
}

template <bool ReLUFused = false>
Tensor _add_scalar_out(Tensor& out, const Tensor& self, Scalar other) {
  int64_t zero_point = out.q_zero_point();
  double scale = out.q_scale();
  int64_t self_zero_point = self.q_zero_point();
  double self_scale = self.q_scale();

  auto iter = TensorIterator::unary_op(out, self);
  AT_DISPATCH_QINT_TYPES(out.scalar_type(), "qadd", [&]() {
    cpu_kernel(iter, [&](scalar_t a) -> scalar_t {
      const auto da = at::dequantize_val(self_scale, self_zero_point, a);
      double c = da + other.toFloat();
      auto quant_val = at::quantize_val<scalar_t>(scale, zero_point, c);
      auto dequant_val = at::dequantize_val(scale, zero_point, quant_val);
      if (ReLUFused) {
        c = std::max<float>(c, 0.0);
      }
      return at::quantize_val<scalar_t>(scale, zero_point, c);
    });
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
  Tensor operator()(Tensor qa, Scalar b,
                    double scale, int64_t zero_point) {
  TORCH_CHECK(qa.qscheme() == kPerTensorAffine ||
              qa.qscheme() == kPerTensorSymmetric,
              "Only per tensor quantization is suuported in Add.");
    auto qc = at::_empty_affine_quantized(qa.sizes(),
      at::device(kCPU).dtype(
        qa.scalar_type()),
        scale,
        zero_point,
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
.op("quantized::add_scalar(Tensor qa, Scalar b, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAddScalar</*ReLUFused=*/false>>(TensorTypeId::QuantizedCPUTensorId))
.op("quantized::add_scalar_relu(Tensor qa, Scalar b, float scale,"
     "int zero_point) -> Tensor qc",
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

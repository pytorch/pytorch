#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>

namespace at { namespace native {
namespace {
template <bool ReLUFused = false>
class QAddInt8 final : public c10::OperatorKernel {
 public:
  Tensor operator()(at::Tensor qa, at::Tensor qb,
                    double scale, int64_t zero_point) {
    TORCH_CHECK(
        qa.numel() == qb.numel(), "Add operands must be the same size!");
    TORCH_CHECK(qa.scalar_type() == qb.scalar_type(), "Add operands should have same data type.");
    auto a = qa.dequantize();
    auto b = qb.dequantize();
    auto c = at::empty_like(a);
    auto iter = TensorIterator::binary_op(c, a, b);

    if (ReLUFused) {
      cpu_kernel(*iter, [&](float a_val, float b_val) -> float {
        return std::max<float>(a_val + b_val, 0);
      });
    } else {
      cpu_kernel(*iter, [&](float a_val, float b_val) -> float {
        return a_val + b_val;
      });
    }
    return at::quantize_linear(c, scale, zero_point, qa.scalar_type());  // Requantize
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::add(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAddInt8</*ReLUFused=*/false>>(QuantizedCPUTensorId()))
.op("quantized::add_relu(Tensor qa, Tensor qb, float scale, int zero_point)"
     "-> Tensor qc",
    c10::RegisterOperators::options()
      .kernel<QAddInt8</*ReLUFused=*/true>>(QuantizedCPUTensorId()));
}  // namespace
}}  // namespace at::native

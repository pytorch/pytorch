#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>

#include <algorithm>
#include <vector>

namespace at { namespace native {
namespace {
template <bool ReLUFused = false>
class QCat final : public c10::OperatorKernel {
 public:
  Tensor operator()(const std::vector<Tensor>& qxs, long axis,
                    double scale, long zero_point) {
    const auto x_dtype = qxs[0].scalar_type();
    std::vector<Tensor> xs;
    xs.reserve(qxs.size());
    for (const auto& qx: qxs) {
      xs.push_back(qx.dequantize());
    }
    const Tensor y = at::cat(xs, axis);
    Tensor qy;
    AT_DISPATCH_QINT_TYPES(x_dtype, "qcat", [&]() {
      qy = at::quantize_linear(y, scale, zero_point, SCALAR_TYPE);
      if (ReLUFused) {
        auto iter = TensorIterator::unary_op(qy, qy);
        unary_kernel(*iter, [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, zero_point));
        });
      }
    });
    return qy;
  }
};

static auto registry = c10::RegisterOperators()
.op("quantized::cat(Tensor[] qx, int axis, float scale, int zero_point)"
    " -> Tensor",
    c10::RegisterOperators::options()
      .kernel<QCat<false>>(QuantizedCPUTensorId()))
.op("quantized::cat_relu(Tensor[] qx, int axis, float scale, int zero_point)"
    " -> Tensor",
    c10::RegisterOperators::options()
      .kernel<QCat<true>>(QuantizedCPUTensorId()));

}  // namespace
}}  // namespace at::native

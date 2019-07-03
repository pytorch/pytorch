#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <algorithm>
#include <vector>

namespace at { namespace native {
namespace {
template <bool ReLUFused = false>
class QCat final : public torch::OperatorKernel {
 public:
  Tensor operator()(const std::vector<Tensor>& qxs, int64_t axis,
                    c10::optional<double> scale,
                    c10::optional<int64_t> zero_point) {
    const auto x_dtype = qxs[0].scalar_type();
    double _scale = scale ? *scale : qxs[0].q_scale();
    int64_t _zero_point = zero_point ? *zero_point : qxs[0].q_zero_point();

    std::vector<Tensor> xs;
    xs.reserve(qxs.size());
    for (const auto& qx: qxs) {
      TORCH_CHECK(x_dtype == qx.scalar_type(),
                   "All dtypes must be the same.");
      xs.push_back(qx.dequantize());
    }
    const Tensor y = at::cat(xs, axis);
    Tensor qy;
    AT_DISPATCH_QINT_TYPES(x_dtype, "qcat", [&]() {
      qy = at::quantize_linear(y, _scale, _zero_point, SCALAR_TYPE);
      if (ReLUFused) {
        auto iter = TensorIterator::unary_op(qy, qy);
        cpu_kernel(*iter, [&](scalar_t value) -> scalar_t {
          return scalar_t(std::max<underlying_t>(value.val_, _zero_point));
        });
      }
    });
    return qy;
  }
};

static auto registry = torch::RegisterOperators()
.op("quantized::cat(Tensor[] qx, int axis, float? scale, int? zero_point)"
    " -> Tensor",
    torch::RegisterOperators::options()
      .kernel<QCat<false>>(QuantizedCPUTensorId()))
.op("quantized::cat_relu(Tensor[] qx, int axis, float? scale, int? zero_point)"
    " -> Tensor",
    torch::RegisterOperators::options()
      .kernel<QCat<true>>(QuantizedCPUTensorId()));

}  // namespace
}}  // namespace at::native

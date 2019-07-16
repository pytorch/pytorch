#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <algorithm>
#include <vector>

namespace at { namespace native {
namespace {

bool is_valid_quantization_type(Tensor t) {
  const auto qtype = t.qscheme();
  return (qtype == kPerTensorAffine) ||
         (qtype == kPerTensorSymmetric);
}

template <bool ReLUFused = false>
class QCat final : public torch::OperatorKernel {
 public:
  Tensor operator()(const std::vector<Tensor>& qxs, int64_t axis,
                    c10::optional<double> scale,
                    c10::optional<int64_t> zero_point) {
    TORCH_CHECK(is_valid_quantization_type(qxs[0]),
                "Only per-tensor quantization is supported in 'cat'!")
    const auto x_dtype = qxs[0].scalar_type();
    const auto x_scale = qxs[0].q_scale();
    const auto x_zero_point = qxs[0].q_zero_point();
    double _scale = scale ? *scale : qxs[0].q_scale();
    int64_t _zero_point = zero_point ? *zero_point : qxs[0].q_zero_point();

    std::vector<Tensor> xs;
    xs.reserve(qxs.size());
    for (const auto& qx: qxs) {
      TORCH_CHECK(is_valid_quantization_type(qx),
                  "Only per-tensor quantization is supported in 'cat'!")
      TORCH_CHECK(x_dtype == qx.scalar_type(), "All dtypes must be the same.");
      TORCH_CHECK(x_scale == qx.q_scale(), "All scales must be the same.");
      TORCH_CHECK(x_zero_point == qx.q_zero_point(),
                  "All zero_points must be the same.");
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

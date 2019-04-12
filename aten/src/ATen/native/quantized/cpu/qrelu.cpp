#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>

#include <algorithm>
#include <tuple>

namespace at { namespace native {
namespace {

class QReluInt8 final : public c10::OperatorKernel {
 public:
  // The use of `double` and `int64_t` is dictated by the
  // `op_registration/infer_schema.h`.
  std::tuple<at::Tensor, double, int64_t>
  operator()(const at::Tensor& x, double scale, int64_t zero_point) {
    at::Tensor tensor = at::empty_like(x);
    auto x_contig = x.contiguous();
    const auto* x_data = x_contig.data<uint8_t>();
    for (int idx = 0; idx < x.numel(); ++idx) {
      tensor[idx] = static_cast<uint8_t>(std::max<int>(x_data[idx],
                                                       zero_point));
    }
    return std::make_tuple(tensor, scale, zero_point);
  }
};

static auto registry = c10::RegisterOperators().op(
    // "quant::relu(x: Tensor, scale: float, zero_point: int)
    //     -> ((Tensor, float, int))"
    c10::FunctionSchema(
        "quantized::relu",
        "",
        std::vector<c10::Argument>{c10::Argument("x", TensorType::get()),
                                   c10::Argument("scale", FloatType::get()),
                                   c10::Argument("zero_point", IntType::get())},
        std::vector<c10::Argument>{
          c10::Argument("y", TensorType::get()),
          c10::Argument("scale_o", FloatType::get()),
          c10::Argument("zero_point_o", IntType::get())
        }),
    c10::kernel<QReluInt8>(),
    c10::dispatchKey(CPUTensorId()));
} // namespace
}} // namespace at::native

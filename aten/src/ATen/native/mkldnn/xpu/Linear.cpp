#include "Linear.h"
// #include "utils/CustomOperatorRegistration.h"

namespace at {
namespace xpu {

using namespace impl;
#define IPEX_LINEAR_DEFINATION(func)                                       \
  Tensor linear_##func(                                                    \
      const Tensor& input,                                                 \
      const Tensor& weight,                                                \
      const c10::optional<Tensor>& bias) {                                 \
    RECORD_FUNCTION(                                                       \
        "linear_" #func, std::vector<c10::IValue>({input, weight, bias})); \
    auto linear_wrapper = LinearConverter();                               \
    auto post_op = [=]() {                                                 \
      xpu::oneDNN::Attr attr;                                                           \
      attr.append_post_eltwise(                                            \
          /* scale */ 1.f,                                                 \
          /* alpha */ 0.f,                                                 \
          /* beta */ 0.f,                                                  \
          attr.kind_with_##func);                                          \
      return attr;                                                         \
    };                                                                     \
    Tensor result;                                                         \
    return linear_wrapper.call(result, input, weight, bias, post_op);      \
  }

#define IPEX_LINEAR_BINARY_DEFINATION(func)                                \
  Tensor linear_binary_##func(                                             \
      const Tensor& input,                                                 \
      const Tensor& weight,                                                \
      const c10::optional<Tensor>& bias,                                   \
      const Tensor& binary) {                                              \
    RECORD_FUNCTION(                                                       \
        "linear_binary_" #func,                                            \
        std::vector<c10::IValue>({input, weight, bias}));                  \
    auto linear_wrapper = LinearConverter();                               \
    auto post_op = [=]() {                                                 \
      xpu::oneDNN::Attr attr;                                                           \
      attr.append_scale_binary(attr.kind_with_binary_##func, binary, 1.f); \
      return attr;                                                         \
    };                                                                     \
    Tensor result;                                                         \
    result = linear_wrapper.call(result, input, weight, bias, post_op);    \
    if (!linear_wrapper.is_fused()) {                                      \
      result = at::func(result, binary);                                   \
    }                                                                      \
    return result;                                                         \
  }


Tensor linear_xpu(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias) {
  auto post_op = [=]() {
    onednn::Attr attr;
    return attr;
  };
  auto linear_wrapper = LinearConverter();
  Tensor result;
  return linear_wrapper.call(result, input, weight, bias, post_op);
}

} // namespace xpu
} // namespace at

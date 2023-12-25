#include "Linear.h"

namespace at {
namespace native::xpu {

using namespace impl;

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

} // namespace native::xpu
} // namespace at

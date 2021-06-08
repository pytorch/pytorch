#include <torch/script.h>

#include "quantized_ops.h"
#include "simple_ops.h"

int main() {
  c10::InferenceMode guard;
  auto input = torch::empty({1, 3, 224, 224});
  at::call_AA_op(input);
  at::call_BB_op(input);
  at::call_CC_op(input);
  at::call_DD_op(input);
  at::call_EE_op(input);
  at::call_FF_op(input);
  const auto t_add = c10::Dispatcher::singleton().findSchemaOrThrow("quantized::t_add", "").typed<at::Tensor(at::Tensor, at::Tensor, double, int64_t)>();
  const auto t_add_relu = c10::Dispatcher::singleton().findSchemaOrThrow("quantized::t_add_relu", "").typed<at::Tensor (at::Tensor, at::Tensor, double, int64_t)>();
  t_add.call(input, input, 1.0, 0);
  t_add_relu.call(input, input, 1.0, 0);
  return 0;
}

#include <torch/script.h>

#include "quantized_ops.h"
#include "simple_ops.h"

int main() {
  torch::autograd::AutoGradMode guard(false);
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  auto input = torch::empty({1, 3, 224, 224});
  at::call_AA_op(input);
  at::call_BB_op(input);
  at::call_CC_op(input);
  at::call_DD_op(input);
  at::call_EE_op(input);
  at::call_FF_op(input);
  callOp("quantized::t_add", "", input, input, 1.0, 0);
  callOp("quantized::t_add_relu", "", input, input, 1.0, 0);
  return 0;
}

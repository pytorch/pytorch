#include <torch/script.h>

#include "op.h"

#include <cstddef>
#include <string>

torch::List<torch::Tensor> custom_op(
    torch::Tensor tensor,
    double scalar,
    int64_t repeat) {
  torch::List<torch::Tensor> output;
  output.reserve(repeat);
  for (int64_t i = 0; i < repeat; ++i) {
    output.push_back(tensor * scalar);
  }
  return output;
}

int64_t custom_op2(std::string s1, std::string s2) {
  return s1.compare(s2);
}

struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor var1, int64_t mul, torch::Tensor var2) {
    ctx->saved_data["mul"] = mul;
    ctx->save_for_backward({var1, var2});
    return var1 + mul*var2 + var1*var2;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_output) {
    int mul = ctx->saved_data["mul"].toInt();
    auto saved = ctx->get_saved_variables();
    auto var1 = saved[0];
    auto var2 = saved[1];
    torch::autograd::variable_list output = {grad_output[0] + grad_output[0]*var2, torch::Tensor(), grad_output[0] * mul + grad_output[0] * var1};
    return output;
  }
};

torch::Tensor custom_op_with_autograd(torch::Tensor var1, int64_t mul, torch::Tensor var2) {
  return CustomOpAutogradFunction::apply(var1, mul, var2);
}

static auto registry =
    torch::RegisterOperators()
        // We parse the schema for the user.
        .op("custom::op", &custom_op)
        .op("custom::op2", &custom_op2)

        // User provided schema. Among other things, allows defaulting values,
        // because we cannot infer default values from the signature. It also
        // gives arguments meaningful names.
        .op("custom::op_with_defaults(Tensor tensor, float scalar = 1, int repeat = 1) -> Tensor[]",
            &custom_op)

        .op("custom::op_with_autograd(Tensor var1, int mul, Tensor var2) -> Tensor",
            &custom_op_with_autograd);

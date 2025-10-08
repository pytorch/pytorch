#include <c10/util/irange.h>
#include <torch/csrc/autograd/functions/utils.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <sstream>
#include <utility>

namespace torch::autograd {

variable_list wrap_outputs(
    const variable_list& inputs,
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    tensor_list&& outputs,
    const function_constructor& ctr) {
  variable_list result;
  result.reserve(outputs.size());
  if (!any_variable_requires_grad(inputs)) {
    for (auto& output : outputs) {
      if (output.defined()) {
        result.push_back(
            make_variable(std::move(output), /*requires_grad=*/false));
      } else {
        result.emplace_back();
      }
    }
  } else {
    auto grad_fn =
        ctr(GradMode::is_enabled() ? collect_next_edges(inputs) : edge_list());
    for (auto& output : outputs) {
      if (output.defined()) {
        auto variable =
            autograd::make_variable(std::move(output), /*requires_grad=*/false);
        autograd::create_gradient_edge(variable, grad_fn);
        result.push_back(std::move(variable));
      } else {
        grad_fn->add_input_metadata(Node::undefined_input());
        result.emplace_back();
      }
    }
  }
  return result;
}

void check_input_variables(
    const char* name,
    const variable_list& inputs,
    int args,
    int required_args,
    bool allow_undefined) {
  if (required_args == -1) {
    required_args = args;
  }
  TORCH_CHECK(
      inputs.size() == static_cast<size_t>(args),
      name,
      ": expected ",
      args,
      " arguments (got ",
      inputs.size(),
      ")");

  for (const auto i : c10::irange(required_args)) {
    TORCH_CHECK(
        inputs[i].defined() || allow_undefined,
        name,
        ": expected Tensor at argument ",
        i,
        " (got None)");
  }
}
} // namespace torch::autograd

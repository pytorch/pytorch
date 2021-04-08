#include <torch/csrc/autograd/functions/utils.h>

#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>

#include <sstream>
#include <vector>

namespace torch { namespace autograd {

variable_list wrap_outputs(const variable_list& inputs, tensor_list&& outputs,
                           const function_constructor& ctr) {
  variable_list result;
  result.reserve(outputs.size());
  if (!any_variable_requires_grad(inputs)) {
    for (auto& output : outputs) {
      if (output.defined()) {
        result.push_back(make_variable(output, /*requires_grad=*/false));
      } else {
        result.emplace_back();
      }
    }
  } else {
    auto grad_fn = ctr(GradMode::is_enabled() ? collect_next_edges(inputs) : edge_list());
    for (auto& output : outputs) {
      if (output.defined()) {
        auto variable = autograd::make_variable(output, /*requires_grad=*/false);
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

void check_input_variables(const char* name, const variable_list& inputs, int args, int required_args, bool allow_undefined) {
  if (required_args == -1) {
    required_args = args;
  }
  if (inputs.size() != (size_t)args) {
    std::stringstream ss;
    ss << name << ": expected " << args << " arguments (got " << inputs.size();
    ss << ")";
    throw std::runtime_error(ss.str());
  }
  for (int i = 0; i < required_args; ++i) {
    if (!inputs[i].defined() && !allow_undefined) {
      std::stringstream ss;
      ss << name << ": expected Tensor at argument " << i << " (got None)";
      throw std::runtime_error(ss.str());
    }
  }
}
}} // namespace torch::autograd

#include "function.h"

#include <string>

#include "variable.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace autograd {

auto Function::flags(const variable_list& inputs) -> FunctionFlags {
  int num_inputs = inputs.size();
  FunctionFlags f;
  f.is_executable = false;
  f.is_volatile = false;
  f.next_functions.resize(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    auto& var = inputs[i];
    if (var) {
      f.is_executable |= var->requires_grad;
      f.is_volatile |= var->is_volatile;
      if (var->grad_fn) {
        f.next_functions[i] = std::make_pair<>(var->grad_fn, var->output_nr);
      } else {
        f.next_functions[i] = std::make_pair<>(var->get_grad_accumulator(), 0);
      }
    }
  }
  f.is_executable &= !f.is_volatile;
  return f;
}

auto Function::name() -> std::string {
  return std::string(typeid(*this).name());
}

void Function::createTrace(const variable_list& inputs, const variable_list& outputs) {
  using namespace torch::jit;
  auto state = tracer::getTracingState(inputs);
  auto& graph = state->graph;
  auto* this_node = graph->appendNewNode<CppOp>(getSharedPtr());
  for (auto& input: inputs) {
    this_node->addInput(tracer::getValueTrace(state, input));
  }
  int num_outputs = outputs.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    Node* sel = graph->appendNewNode<Select>(this_node, i);
    sel->inferTypeFrom(output->data);
    tracer::setValueTrace(state, output, sel);
  }
}

}} // namespace torch::autograd

#include "function.h"

#include <string>

#include "variable.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/functions/tensor.h"

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

// This function is analogous to make_trace which operates on PythonOp, but this
// function instead works for C++ implemented autograd Functions, which don't
// actually have any backing Python class. We still need to trace them!
variable_list Function::tracedApply(variable_list inputs) {
  using namespace torch::jit;
  bool is_traceable = static_cast<bool>(dynamic_cast<Identity*>(this));
  // Traceable Functions are completely transparent to the JIT.
  if (is_traceable) {
    return apply(inputs);
  }
  auto state = tracer::getTracingState(inputs);

  // Register eval hooks if backward of this function is not traceable.
  // This has to be done early, because it modifies inputs.
  bool is_backward_traceable = false;
  std::shared_ptr<tracer::EvalCommonState> eval_state;
  if (!is_backward_traceable) {
    eval_state = tracer::EvalExitHook::registerHook(state, inputs);
  }

  // Insert a CppOp in the trace.
  auto& graph = state->graph;
  auto* this_node = graph->create<CppOp>(getSharedPtr());
  for (auto& input: inputs) {
      this_node->addInput(tracer::getValueTrace(state, input));
  }
  graph->appendNode(this_node);

  // Finally apply this Function.
  variable_list outputs = apply(inputs);

  // Set up output traces.
  int num_outputs = outputs.size();
  for (int i = 0; i < num_outputs; ++i) {
      auto& output = outputs[i];
      Node* sel = graph->appendNode(graph->create<Select>(this_node, i));
      sel->inferTypeFrom(output->data);
      tracer::setValueTrace(state, output, sel);
  }

  // Register the point where Eval region starts in backward.
  // NOTE: this modifies outputs.
  if (!is_backward_traceable) {
    tracer::EvalEnterHook::registerHook(state, outputs, eval_state);
  }
  return outputs;
}

}} // namespace torch::autograd

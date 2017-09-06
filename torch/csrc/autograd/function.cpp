#include "function.h"

#include <string>

#include "variable.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/autograd/functions/special.h"

namespace torch { namespace autograd {

auto Function::flags(const variable_list& inputs) -> FunctionFlags {
  int num_inputs = inputs.size();
  FunctionFlags f;
  f.is_executable = false;
  f.is_volatile = false;
  f.next_functions.resize(num_inputs);
  for (int i = 0; i != num_inputs; ++i) {
    if (inputs[i].defined()) {
      auto& var = inputs[i];
      f.is_executable |= var.requires_grad();
      f.is_volatile |= var.is_volatile();
      if (var.grad_fn()) {
        f.next_functions[i] = std::make_pair<>(var.grad_fn(), var.output_nr());
      } else {
        f.next_functions[i] = std::make_pair<>(var.grad_accumulator(), 0);
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
  // Traceable Functions are completely transparent to the JIT.
  if (is_traceable()) {
    return apply(inputs);
  }
  auto state = tracer::getTracingState(inputs);
  auto state_lock = state->lock();

  // Insert a CppOp in the trace.
  auto& graph = state->graph;
  auto* this_node = graph->createCppOp(getSharedPtr());
  for (auto& input: inputs) {
    this_node->addInput(tracer::getValueTrace(state, input));
  }
  graph->appendNode(this_node);

  // Finally apply this Function.
  state_lock.unlock();
  variable_list outputs = apply(inputs);
  state_lock.lock();

  // Set up output traces.
  int num_outputs = outputs.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    if (!output) {
      auto & edge = next_functions[i];
      auto & meta = edge.first->input_sizes.at(edge.second);
      output = meta.recreate();
    }
    Node* sel = graph->appendNode(graph->createSelect(this_node, i));
    sel->inferTypeFrom(output.data());
    tracer::setValueTrace(state, output, sel);
  }

  if (!passes_state_transparently()) {
    auto this_eval = dynamic_cast<Eval*>(this);
    // Evals consume handle from a context edge of forward node
    if (this_eval)
      this_node->addInput(this_eval->forward_ctx_select);
    // There's no point in wrapping functions in Eval, if we know they already are
    // part of another Eval subgraph. This is both a small optimization, and
    // it allows us to not implement saved_variables() in many functions.
    bool should_trace_backward = tracing_state->in_eval_subgraph;
    if (!should_trace_backward) {
      auto saved_vars = saved_variables();
      if (!saved_vars)
        throw std::runtime_error(std::string("saved_variables() needed but not implemented in ") + name());
      variable_list bw_subgraph_inputs(inputs);
      for (auto& saved_var : *saved_vars) {
        bw_subgraph_inputs.emplace_back(saved_var.unpack(getSharedPtr()));
      }
      tracer::nontraceableBackwardSubgraph(bw_subgraph_inputs, outputs);
    }
    bool has_backwards_eval = !should_trace_backward || this_eval;
    if (has_backwards_eval)
      setUpContextEdge(this_node, num_outputs, inputs, outputs);
  }
  return outputs;
}

void Function::setUpContextEdge(jit::Node* node, int ctx_output_nr,
                                const variable_list& inputs, const variable_list& outputs) {
  jit::Graph* graph = node->owningGraph();
  jit::Node* ctx_select = graph->appendNode(graph->createSelect(node, ctx_output_nr));
  ctx_select->setType(std::make_shared<jit::HandleType>());
  auto backward_eval = Eval::getBackwardEval(inputs, outputs);
  if (backward_eval)
    backward_eval->forward_ctx_select = ctx_select;
}

}} // namespace torch::autograd

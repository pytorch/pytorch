#include <Python.h>

#include "torch/csrc/autograd/function.h"

#include "torch/csrc/autograd/functions/special.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace autograd {

thread_local uint64_t Function::next_sequence_nr_ = 0;

auto Function::name() -> std::string {
  return std::string(typeid(*this).name());
}

// This function is analogous to make_trace which operates on PythonOp, but this
// function instead works for C++ implemented autograd Functions, which don't
// actually have any backing Python class. We still need to trace them!
variable_list Function::traced_apply(variable_list inputs) {
  using namespace torch::jit;
  // Traceable Functions are completely transparent to the JIT.
  if (is_traceable()) {
    return apply(inputs);
  }
  auto state = tracer::getTracingState(inputs);
  auto state_lock = state->lock();

  // Insert a CppOp in the trace.
  auto& graph = state->graph;
  std::vector<VariableFlags> var_flags;
  for(auto & input: inputs) {
    var_flags.push_back(VariableFlags::of(input));
  }
  auto* this_node = graph->createCppOp(get_shared_ptr(), std::move(var_flags));
  this_node->setSourceLocation(std::make_shared<StringSourceLocation>(
        jit::tracer::getPythonInterpreterStackTrace()
  ));
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
    auto sel = this_node->addOutput();
    // TODO: At the moment, C++ does not track shared storage.  It
    // should.  Update this when that happens.
    if (output.defined()) {
      sel->inferTypeFrom(output.data());
      tracer::setValueTrace(state, output, sel);
    }
  }

  if (!passes_state_transparently()) {
    auto this_eval = dynamic_cast<Eval*>(this);
    // Evals consume handle from a context edge of forward node
    if (this_eval)
      this_node->addInput(this_eval->forward_ctx_select);
    // There's no point in wrapping functions in Eval, if we know they already are
    // part of another Eval subgraph. This is both a small optimization, and
    // it allows us to not implement saved_variables() in many functions.
    const bool should_trace_backward = tracing_state_->in_eval_subgraph;
    if (!should_trace_backward) {
      auto saved_vars = saved_variables();
      if (!saved_vars)
        throw std::runtime_error("saved_variables() needed but not implemented in " + name());
      variable_list bw_subgraph_inputs(inputs);
      for (auto& saved_var : *saved_vars) {
        bw_subgraph_inputs.emplace_back(saved_var.unpack(get_shared_ptr()));
      }
      tracer::nontraceableBackwardSubgraph(bw_subgraph_inputs, outputs);
    }
    bool has_backwards_eval = !should_trace_backward || this_eval;
    if (has_backwards_eval)
      set_up_context_edge(this_node, inputs, outputs);
  }
  return outputs;
}

void Function::set_up_context_edge(
    jit::Node* this_node,
    const variable_list& inputs,
    const variable_list& outputs) {
  auto ctx_select = this_node->addOutput();
  ctx_select->setType(jit::HandleType::get());
  auto backward_eval = Eval::getBackwardEval(inputs, outputs);
  if (backward_eval)
    backward_eval->forward_ctx_select = ctx_select;
}

}} // namespace torch::autograd

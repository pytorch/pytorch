#include "interpreter_autograd_function.h"

namespace torch { namespace jit {

using namespace torch::jit::tracer;

autograd::variable_list InterpreterAutogradFunction::apply(
    const autograd::variable_list& inputs) {
  // Initial correctness checks.
  if (stage_ == stage_details_.size()) {
    throw std::runtime_error(std::string("Function compiled only for ") +
        std::to_string(stage_details_.size() - 1) + " derivatives. Use nderivs argument " +
        "to request more.");
  }
  if (used_) throw std::runtime_error(autograd::ERR_BACKWARD_TWICE);
  used_ |= !keep_graph_;

  const auto & details = stage_details_[stage_];

  // Validate inputs
  for (std::size_t i = 0; i < (std::size_t)num_inputs; ++i) {
    if (!details.input_flags[i].verify(inputs[i])) {
      throw std::runtime_error("JIT interpreter received inputs with different "
          "flags than it was compiled for.");
    }
  }

  // Run the interpreter
  auto tinputs = fmap(inputs, [](const autograd::Variable& i) { return i.data(); });
  std::vector<at::Tensor> toutputs;
  InterpreterState interp = (keep_graph_) ? interp_.clone() : interp_;
  interp.runOneStage(tinputs, toutputs);

  // Lazily create grad_fn
  std::shared_ptr<Function> grad_fn;
  auto make_grad_fn = [&]() {
    grad_fn = std::make_shared<InterpreterAutogradFunction>(
        std::move(interp), stage_details_, stage_ + 1);
    // Patch next_functions to include prevous stage next_functions
    // This is needed because stage N is really a derivative of
    // all stages from 1 to N-1. If a part of stage x graph is
    // reused in stage y (y > x), it is inlined by the tracer,
    // and so we need to copy next_fns because those Variables
    // aren't real inputs to that stage, so that's the only place
    // where we can get them.
    for (auto copied_idx : details.copied_next_fns) {
      grad_fn->next_functions.push_back(next_functions[copied_idx]);
    }
    // Add grad_fns corresponding to inputs
    for (auto & input : inputs) {
      if (!input.requires_grad()) continue; // See Note [Null-edge pruning]
      grad_fn->next_functions.emplace_back(
        input.grad_fn() ? input.grad_fn() : input.grad_accumulator(),
        input.output_nr());
    }
  };

  // Wrap the outputs
  // TODO: handle views
  autograd::variable_list result;
  JIT_ASSERT(toutputs.size() == details.output_flags.size());
  auto num_outputs = toutputs.size();
  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto & flags = details.output_flags[i];
    if (flags.requires_grad) { // See Note [Null-edge pruning]
      if (!grad_fn) make_grad_fn();
      result.push_back(autograd::make_variable(toutputs[i], grad_fn));
    } else {
      result.push_back(autograd::make_variable(toutputs[i], false, flags.is_volatile));
    }
  }

  return result;
}

InterpreterFunctionFactory::InterpreterFunctionFactory(TracingState *state) {
  code_ = jit::Code(state->graph);
  stage_details_.resize(state->graph->stage() + 1);
  for (std::size_t stage = 0; stage < state->graph->stage() + 1; ++stage) {
    auto & details = stage_details_[stage];
    std::tie(details.input_flags, details.output_flags) = std::move(state->var_flags[stage]);
    if (stage >= 1) {
      auto & current_outputs = state->output_edges[stage];
      auto & prev_outputs = state->output_edges[stage - 1];
      for (auto & output : current_outputs) {
        // Check if output appears in outputs of previous stage
        auto prev_it = std::find(prev_outputs.begin(), prev_outputs.end(), output);
        if (prev_it == prev_outputs.end()) continue;
        // If yes, find its index and append that to the list of edges that will need
        // to be copied in InterpreterAutogradFunction.
        details.copied_next_fns.push_back(std::distance(prev_outputs.begin(), prev_it));
      }
    }
  }
}

std::shared_ptr<InterpreterAutogradFunction> InterpreterFunctionFactory::construct() {
  return std::make_shared<InterpreterAutogradFunction>(code_, stage_details_);
}

}}

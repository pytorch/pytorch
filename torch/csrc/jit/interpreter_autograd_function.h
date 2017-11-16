#pragma once

#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"

namespace torch { namespace jit {

struct StageDetails {
  std::vector<tracer::VariableFlags> input_flags;
  std::vector<tracer::VariableFlags> output_flags;
  std::vector<int> copied_next_fns;
};

struct InterpreterAutogradFunction : public autograd::Function {
  InterpreterAutogradFunction(const jit::Code & code,
                              const std::vector<StageDetails>& stage_details)
    : interp_(code)
    , stage_details_(stage_details)
    , stage_(0) {}

  InterpreterAutogradFunction(InterpreterState interp,
                              const std::vector<StageDetails>& stage_details,
                              std::size_t stage)
    : interp_(std::move(interp))
    , stage_details_(stage_details)
    , stage_(stage) {}

  virtual void willReleaseVariables() override {
    keep_graph_ = false;
  }

  virtual autograd::variable_list apply(const autograd::variable_list& inputs) override {
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
      grad_fn->is_executable = true;
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

private:
  InterpreterState interp_;
  const std::vector<StageDetails>& stage_details_;
  size_t stage_;
  bool keep_graph_ = true;
  bool used_ = false;
};

}}

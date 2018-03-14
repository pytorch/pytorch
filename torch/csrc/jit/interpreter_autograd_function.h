#pragma once

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/variable_flags.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace torch { namespace jit {
namespace tracer {
struct TracingState;
} // namespace tracer

struct StageDetails {
  std::vector<VariableFlags> input_flags;
  std::vector<VariableFlags> output_flags;
  std::vector<int> copied_next_fns;
  std::vector<bool> used_inputs;
};

struct InterpreterAutogradFunction : autograd::Function {
  InterpreterAutogradFunction(
      const jit::Code& code,
      const std::vector<StageDetails>& stage_details)
      // Stage 0 isn't run through the autograd, so we set this
      // here just in case it is used.
      : Function(/*num_inputs=*/stage_details.at(0).input_flags.size()),
        interp_(code),
        stage_details_(stage_details),
        stage_(0) {}

  InterpreterAutogradFunction(InterpreterState interp,
                              const std::vector<StageDetails>& stage_details,
                              std::size_t stage)
    : interp_(std::move(interp))
    , stage_details_(stage_details)
    , stage_(stage) {}

  // apply() is a protected method in `autograd::Function` since users should
  // usually use the call operator, which invokes either `apply()` or
  // `traced_apply()` depending on whether the function is traced. For
  // InterpreterAutogradFunctions, however, we don't need this extra tracing
  // logic. So we make it public here.
  using autograd::Function::apply;

  virtual void will_release_variables() override {
    keep_graph_ = false;
  }

  virtual autograd::variable_list apply(const autograd::variable_list& inputs) override;

private:
  InterpreterState interp_;
  const std::vector<StageDetails>& stage_details_;
  size_t stage_;
  bool keep_graph_ = true;
  bool used_ = false;
};

struct InterpreterFunctionFactory {
  explicit InterpreterFunctionFactory(tracer::TracingState *state);
  // Return `InterpreterAutogradFunction` because it has its apply() public.
  std::shared_ptr<InterpreterAutogradFunction> construct();
  // For when we need to pass a function with this signature.
  std::shared_ptr<autograd::Function> construct_function() {
    return construct();
  }

 private:
  jit::Code code_;
  std::vector<StageDetails> stage_details_;
};


}}

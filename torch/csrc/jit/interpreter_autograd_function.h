#pragma once

#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"

namespace torch { namespace jit {

struct StageDetails {
  std::vector<VariableFlags> input_flags;
  std::vector<VariableFlags> output_flags;
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
  std::shared_ptr<autograd::Function> construct();

private:
  jit::Code code_;
  std::vector<StageDetails> stage_details_;
};


}}

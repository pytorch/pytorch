#pragma once

#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
namespace torch { namespace jit {
struct InterpreterAutogradFunction : public autograd::Function {
  InterpreterAutogradFunction(const jit::Code & code)
  : interp_(code) {}
  InterpreterAutogradFunction(const InterpreterState & interp_, autograd::FunctionFlags && f)
  : autograd::Function(std::move(f)), interp_(interp_) {}

  virtual void willReleaseVariables() override {
    keep_graph = false;
  }
  virtual autograd::variable_list apply(const autograd::variable_list& inputs) override {
    std::vector<at::Tensor> tinputs;
    std::vector<at::Tensor> toutputs;
    for(auto & i : inputs) {
      tinputs.push_back(i.data());
    }
    InterpreterState interp = (keep_graph) ? interp_.clone() : interp_;
    keep_graph = true;
    interp.runOneStage(tinputs, toutputs);
    auto r = autograd::wrap_outputs(inputs, std::move(toutputs), [&](autograd::FunctionFlags f) {
      return std::make_shared<InterpreterAutogradFunction>(interp, std::move(f));
    });
    return r;
  }
private:
  bool keep_graph = true;
  InterpreterState interp_;
};

}}

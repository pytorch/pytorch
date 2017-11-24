#pragma once

#include <Python.h>
#include <memory>
#include <unordered_map>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"

namespace torch { namespace autograd {

struct MultiStageClosure;

struct AutogradClosureFactory {
  AutogradClosureFactory(torch::jit::tracer::TracingState *graph);

  std::shared_ptr<Function> construct();

  std::shared_ptr<MultiStageClosure> desc;
};

struct AutogradClosure : public Function {
  AutogradClosure(const std::shared_ptr<MultiStageClosure>& desc);

  virtual variable_list apply(const variable_list& inputs) override;

private:
  AutogradClosure(const std::shared_ptr<MultiStageClosure>& desc, std::size_t stage);

  variable_list rewrapInputs(const variable_list& inputs);

  std::shared_ptr<MultiStageClosure> desc;
  std::size_t stage;

  std::unordered_map<int, at::Tensor> saved_vars;
  std::unordered_map<int, std::shared_ptr<Function>> saved_handles;

  Engine::pre_callback_map pre_callbacks;
  Engine::post_callback_map post_callbacks;

  std::unordered_map<int, at::Tensor> captured_vars;
  std::unordered_map<int, std::shared_ptr<Function>> captured_handles;
  tensor_list outputs;
  std::mutex capture_mutex;
};

}} // namespace torch::autograd

#pragma once

#include <map>
#include <memory>

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/grad_mode.h"

// for AutoGPU. Usage:
//   AutoGPU gpu_raii(1);
// While this object is in scope, all of your GPU tensors will go to GPU 1
#include "torch/csrc/utils/auto_gpu.h"

#define AUTOGRAD_OPTIMIZER_CLASS(Type) \
  class Type : public torch::Optimizer_CRTP<Type>
#define AUTOGRAD_KWARG(CLS, TYP, NAME, DEFAULT, OPTION) \
  TYP NAME##_ = DEFAULT;                                \
  CLS& NAME(TYP x = OPTION) {                           \
    NAME##_ = x;                                        \
    return *this;                                       \
  }

namespace {
namespace tag = torch::autograd;
using IntVec = decltype(std::declval<at::IntList>().vec());
} // namespace

namespace torch {
namespace detail {
extern tag::Engine engine;
}

namespace nn {
class Module;
} // namespace nn

class OptimizerImpl;
using Variable = tag::Variable;
using variable_list = tag::variable_list;
using Tensor = at::Tensor;
using Optimizer = std::shared_ptr<OptimizerImpl>;

void backward(Tensor loss, bool keep_graph = false);

inline Variable Var(at::Tensor data, bool requires_grad = true) {
  return tag::make_variable(data, requires_grad);
}

// This is thread local!!!
inline void set_grad_enabled(bool val = true) {
  tag::GradMode::set_enabled(val);
}

// RAII thread local lock that stops future execution from building gradients
class no_grad_guard {
 public:
  no_grad_guard() {
    tag::GradMode::set_enabled(false);
  }

  ~no_grad_guard() {
    tag::GradMode::set_enabled(true);
  }
};

void setSeed(uint64_t seed);

int getNumGPUs();
bool hasCuda();
bool hasCudnn();

} // namespace torch

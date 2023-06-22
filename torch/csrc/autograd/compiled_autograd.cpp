#include <torch/csrc/autograd/compiled_autograd.h>
#include <torch/csrc/autograd/generated/Functions.h>

namespace torch {
namespace autograd {

void CompiledNodeArgs::collect(torch::autograd::generated::TypeAndSize& t) {
  collect(t.sym_sizes);
  collect(t.options);
}

void SwapSavedVariables::before(torch::autograd::generated::TypeAndSize& t) {
  before(t.sym_sizes);
  before(t.options);
}

void SwapSavedVariables::after(torch::autograd::generated::TypeAndSize& t) {
  after(t.sym_sizes);
  after(t.options);
}

} // namespace autograd
} // namespace torch

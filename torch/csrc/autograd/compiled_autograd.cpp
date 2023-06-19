#include <torch/csrc/autograd/compiled_autograd.h>
#include <torch/csrc/autograd/generated/Functions.h>

namespace torch {
namespace autograd {

void CompiledNodeArgs::collect(torch::autograd::generated::TypeAndSize& t) {
  collect(t.sym_sizes);
  collect(t.options);
}

} // namespace autograd
} // namespace torch

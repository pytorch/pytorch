#include <torch/csrc/jit_decomp_interface.h>

namespace torch {
namespace autograd {
namespace impl {

namespace {
JitDecompInterface* fns = nullptr;
}

void setJitDecompInterface(JitDecompInterface* f) {
  fns = f;
}
JitDecompInterface* getJitDecomp() {
  TORCH_CHECK(
      fns,
      "Support for JIT decompositions has not been loaded; have you linked against TBD?")
  return fns;
}

} // namespace impl
} // namespace autograd
} // namespace torch

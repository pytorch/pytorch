#include <torch/csrc/autograd/jit_decomp_interface.h>

namespace torch {
namespace autograd {
namespace impl {

namespace {
JitDecompInterface* impl = nullptr;
}

void setJitDecompImpl(JitDecompInterface* impl_) {
  impl = impl_;
}

JitDecompInterface* getJitDecompImpl() {
  return impl;
}

} // namespace impl
} // namespace autograd
} // namespace torch

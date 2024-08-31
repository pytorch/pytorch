#include <torch/csrc/autograd/jit_decomp_interface.h>

namespace torch::autograd::impl {

namespace {
JitDecompInterface* impl = nullptr;
}

void setJitDecompImpl(JitDecompInterface* impl_) {
  impl = impl_;
}

JitDecompInterface* getJitDecompImpl() {
  return impl;
}

} // namespace torch::autograd::impl

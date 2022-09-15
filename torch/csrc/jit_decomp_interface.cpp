#include <torch/csrc/jit_decomp_interface.h>

namespace torch {
namespace autograd {
namespace impl {

namespace {
JitDecompInterface* impl = nullptr;
}

void setJitDecompImpl(JitDecompInterface* impl_) {
  impl = impl_;
}

JitDecompInterface* getJitDecompImpl(c10::string_view name) {
  return impl;
}

} // namespace impl
} // namespace autograd
} // namespace torch

#include <ATen/core/VariableFallbackKernel.h>

namespace {

TORCH_LIBRARY_IMPL(_, AutogradOther, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradCPU, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradXPU, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradCUDA, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradMTIA, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradMAIA, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradXLA, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradLazy, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradMPS, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

TORCH_LIBRARY_IMPL(_, AutogradMeta, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

// see Note [ADInplaceOrView key]
TORCH_LIBRARY_IMPL(_, ADInplaceOrView, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, AutogradHPU, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

} // namespace

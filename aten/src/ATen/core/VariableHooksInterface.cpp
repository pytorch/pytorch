#include <ATen/core/VariableHooksInterface.h>

namespace at { namespace impl {

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
VariableHooksInterface* hooks = nullptr;
}

void SetVariableHooks(VariableHooksInterface* h) {
  hooks = h;
}
VariableHooksInterface* GetVariableHooks() {
  TORCH_CHECK(hooks, "Support for autograd has not been loaded; have you linked against libtorch.so?")
  return hooks;
}

}} // namespace at::impl

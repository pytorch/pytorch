#include <ATen/detail/VariableHooksInterface.h>

namespace at { namespace detail {

const VariableHooksInterface& getVariableHooks() {
  static std::unique_ptr<VariableHooksInterface> var_hooks;
  if (!var_hooks) {
    var_hooks = VariableHooksRegistry()->Create("VariableHooks");
    if (!var_hooks) {
      var_hooks = std::unique_ptr<VariableHooksInterface>(new VariableHooksInterface());
    }
  }
  return *var_hooks;
}

AT_DEFINE_REGISTRY(VariableHooksRegistry, VariableHooksInterface);

}} // namespace at::detail

#include <ATen/detail/VariableHooksInterface.h>

namespace at { namespace detail {

// NB: The VariableHooks returned by this function may CHANGE after dlopen()
const VariableHooksInterface& getVariableHooks() {
  static std::unique_ptr<VariableHooksInterface> var_hooks;
  if (!var_hooks) {
    var_hooks = VariableHooksRegistry()->Create("VariableHooks");
  }
  if (var_hooks) {
    return *var_hooks;
  }
  static std::unique_ptr<VariableHooksInterface> default_var_hooks =
    std::unique_ptr<VariableHooksInterface>(new VariableHooksInterface());
  return *default_var_hooks;
}

AT_DEFINE_REGISTRY(VariableHooksRegistry, VariableHooksInterface);

}} // namespace at::detail

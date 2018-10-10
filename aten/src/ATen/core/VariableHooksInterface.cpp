#include <ATen/core/VariableHooksInterface.h>

namespace at {

namespace detail {

  // NB: The VariableHooks returned by this function may CHANGE after dlopen()
  // NB: This function takes a lock, don't call it from perf critical paths
  const VariableHooksInterface& getVariableHooks() {
    static std::mutex var_hooks_mutex;
    static std::unique_ptr<VariableHooksInterface> var_hooks = nullptr;
    static std::unique_ptr<VariableHooksInterface> default_var_hooks =
      std::unique_ptr<VariableHooksInterface>(new VariableHooksInterface());
    std::lock_guard<std::mutex> lock(var_hooks_mutex);

    if (!var_hooks) {
      var_hooks = VariableHooksRegistry()->Create("VariableHooks", VariableHooksArgs{});
    }
    if (var_hooks) {
      return *var_hooks;
    }
    return *default_var_hooks;
  }

}

C10_DEFINE_REGISTRY(
    VariableHooksRegistry,
    VariableHooksInterface,
    VariableHooksArgs)

} // namespace at::detail

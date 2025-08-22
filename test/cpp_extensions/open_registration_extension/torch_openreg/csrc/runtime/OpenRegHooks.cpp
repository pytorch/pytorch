#include "OpenRegHooks.h"

namespace c10::openreg {

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());

  return true;
}();

} // namespace c10::openreg

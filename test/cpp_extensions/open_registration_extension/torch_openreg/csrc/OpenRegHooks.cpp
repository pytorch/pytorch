#include "OpenRegHooks.h"

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());

  return true;
}();

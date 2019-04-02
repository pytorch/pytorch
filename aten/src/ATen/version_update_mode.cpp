#include <ATen/version_update_mode.h>

namespace at {

thread_local bool VersionUpdateMode_enabled = true;

bool VersionUpdateMode::is_enabled() {
  return VersionUpdateMode_enabled;
}

void VersionUpdateMode::set_enabled(bool enabled) {
  VersionUpdateMode_enabled = enabled;
}

}

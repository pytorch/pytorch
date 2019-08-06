#include <c10/core/MemoryFormat.h>
namespace c10 {
namespace {
thread_local bool memory_format_proparation_enabled = false;
}

bool get_memory_format_proparation() {
  return memory_format_proparation_enabled;
}

void set_memory_format_proparation(bool value) {
  memory_format_proparation_enabled = value;
}

} // namespace c10

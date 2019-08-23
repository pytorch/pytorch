#include <c10/core/MemoryFormat.h>
namespace c10 {
namespace {
thread_local bool memory_format_propagation_enabled = false;
}

bool get_memory_format_propagation() {
  return memory_format_propagation_enabled;
}

void set_memory_format_propagation(bool value) {
  memory_format_propagation_enabled = value;
}

} // namespace c10

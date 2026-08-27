#include <c10/util/TypeCast.h>

namespace c10 {

// Export c10::report_overflow as C10_API for libtorch ABI. The throw logic
// lives in torch::headeronly::report_overflow; this wrapper delegates to it.
[[noreturn]] void report_overflow(const char* name) {
  torch::headeronly::report_overflow(name);
}

} // namespace c10

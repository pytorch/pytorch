#include <c10/util/TypeCast.h>

namespace c10 {

[[noreturn]] void report_overflow(const char* name) {
  TORCH_CHECK(
      false,
      "value cannot be converted to type ",
      name,
      " without overflow");
}

} // namespace c10

#include <c10/util/safe_conv.h>

#include <c10/util/Exception.h>

#include <sstream>
#include <utility>

namespace c10::detail {

[[noreturn]] void report_narrowing_overflow(const char* name) {
  std::ostringstream oss;
  oss << "value cannot be safely converted without overflow";
  if (name != nullptr) {
    oss << ": " << name;
  }
  TORCH_CHECK(false, std::move(oss).str());
}

} // namespace c10::detail

#include <c10/util/safe_conv.h>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace c10::detail {

[[noreturn]] void report_narrowing_overflow(const char* name) {
  std::ostringstream oss;
  oss << "value cannot be safely converted without overflow";
  if (name != nullptr) {
    oss << ": " << name;
  }
  throw std::runtime_error(std::move(oss).str());
}

} // namespace c10::detail

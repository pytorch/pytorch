#include <c10/util/safe_conv.h>

#include <stdexcept>

#include <sstream>
#include <utility>

namespace c10::detail {

[[noreturn]] void report_narrowing_overflow(const char* name) {
  std::ostringstream oss;
  if (name == nullptr) {
    oss << "value cannot be safely converted without overflow";
  } else {
    oss << "value cannot be converted to type " << name << " without overflow";
  }
  // @allow-raw-throw: preserve the checked-conversion exception contract.
  throw std::runtime_error(std::move(oss).str());
}

} // namespace c10::detail

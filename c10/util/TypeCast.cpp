#include <c10/util/TypeCast.h>

#include <c10/util/Exception.h>

namespace c10 {

[[noreturn]] void report_overflow(const char* name) {
  std::ostringstream oss;
  oss << "value cannot be converted to type " << name << " without overflow";
  // c10::Error rather than domain_error (issue 33562)
  TORCH_CHECK(false, std::move(oss).str());
}

} // namespace c10

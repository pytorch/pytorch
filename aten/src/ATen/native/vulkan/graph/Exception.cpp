#include <ATen/native/vulkan/graph/Exception.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.func << " at " << loc.file << ": " << loc.line;
  return out;
}

Error::Error(SourceLocation location, std::string msg)
    : location_{location}, msg_(std::move(msg)) {
  refresh_what();
}

void Error::refresh_what() {
  what_ = compute_what(/*include_backtrace =*/true);
}

std::string Error::compute_what(bool include_source) const {
  std::ostringstream oss;
  oss << msg_;

  if (include_source) {
    oss << "\n"
        << "Raised from: " << location_;
  }

  return oss.str();
}

} // namespace vulkan
} // namespace native
} // namespace at

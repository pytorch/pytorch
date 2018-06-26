#include <ATen/Error.h>
#include <ATen/Backtrace.h>

#include <ostream>
#include <string>

namespace at {
std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.function << " at " << loc.file << ":" << loc.line;
  return out;
}

Error::Error(SourceLocation source_location, std::string err)
  : what_without_backtrace_(err)
  , what_(str(err, " (", source_location, ")\n", get_backtrace(/*frames_to_skip=*/2)))
  {}

} // namespace at

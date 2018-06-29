#include <ATen/Error.h>
#include <ATen/Backtrace.h>

#include <iostream>
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

void Warning::warn(SourceLocation source_location, std::string msg) {
  warning_handler_(source_location, msg.c_str());
}

void Warning::set_warning_handler(handler_t handler) {
  warning_handler_ = handler;
}

void Warning::print_warning(const SourceLocation& source_location, const char* msg) {
  std::cerr << "Warning: " << msg << " (" << source_location << ")\n";
}

Warning::handler_t Warning::warning_handler_ = &Warning::print_warning;

} // namespace at

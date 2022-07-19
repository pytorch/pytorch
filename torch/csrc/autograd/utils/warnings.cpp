#include <torch/csrc/autograd/utils/warnings.h>

namespace torch {
namespace autograd {
namespace utils {

void DelayWarningHandler::process(
    const at::SourceLocation& source_location,
    const std::string& msg,
    const bool verbatim) {
  std::lock_guard<std::mutex> lock(mutex_);
  warnings_.push_back({source_location, msg, verbatim});
}

void DelayWarningHandler::replay_warnings() {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      c10::Warning::get_warning_handler() != this,
      "DelayWarningHandler cannot replay warnings into itself, this will cause a deadlock");
  for (const auto& warning : warnings_) {
    c10::Warning::warn(warning.source_location, warning.msg, warning.verbatim);
  }
}

} // namespace utils
} // namespace autograd
} // namespace torch

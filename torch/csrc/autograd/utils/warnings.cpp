#include <torch/csrc/autograd/utils/warnings.h>

namespace torch {
namespace autograd {
namespace utils {

void DelayWarningHandler::process(const c10::Warning& warning) {
  std::lock_guard<std::mutex> lock(mutex_);
  warnings_.push_back(warning);
}

void DelayWarningHandler::replay_warnings() {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(
      c10::WarningUtils::get_warning_handler() != this,
      "DelayWarningHandler cannot replay warnings into itself, this will cause a deadlock");
  for (const auto& warning : warnings_) {
    c10::warn(warning);
  }
}

} // namespace utils
} // namespace autograd
} // namespace torch

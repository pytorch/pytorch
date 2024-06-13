#include <torch/csrc/monitor/instrumentation.h>

#include <chrono>
#include <string_view>

namespace torch {
namespace monitor {

WaitCounterHandle::WaitCounterHandle(std::string_view key)
    : impl_(nullptr) {
  // implement
}

void WaitCounterHandle::start(std::chrono::steady_clock::time_point now) {
  // implement
}

void WaitCounterHandle::stop(std::chrono::steady_clock::time_point now) {
  // implement
}

} // namespace monitor
} // namespace torch

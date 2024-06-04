#include <torch/csrc/monitor/instrumentation.h>

#include <chrono>
#include <string_view>

namespace torch {
namespace monitor {

WaitCounterUs::WaitCounterUs(std::string_view key) : key_(key) {
  // implement
}

WaitCounterUs::~WaitCounterUs() {
  // implement
}

void WaitCounterUs::start(std::chrono::steady_clock::time_point now) {
  // implement
}

void WaitCounterUs::stop(std::chrono::steady_clock::time_point now) {
  // implement
}

} // namespace monitor
} // namespace torch

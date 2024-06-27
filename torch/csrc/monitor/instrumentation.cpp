#ifndef FBCODE_CAFFE2
#include <torch/csrc/monitor/instrumentation.h>

#include <chrono>
#include <string_view>

namespace torch {
namespace monitor {

namespace detail {
class WaitCounterImpl {};
static detail::WaitCounterImpl& getImpl(std::string_view key) {
  auto* impl = new detail::WaitCounterImpl();
  return *impl;
}
} // namespace detail

WaitCounterHandle::WaitCounterHandle(std::string_view key)
    : impl_(detail::getImpl(key)) {
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
#endif

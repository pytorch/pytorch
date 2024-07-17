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

WaitCounterHandle::WaitGuard WaitCounterHandle::start() {
  // implement
  return WaitCounterHandle::WaitGuard(*this, 0);
}

void WaitCounterHandle::stop(intptr_t) {
  // implement
}

} // namespace monitor
} // namespace torch
#endif

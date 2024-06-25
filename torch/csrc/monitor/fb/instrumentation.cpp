#include <data_preproc/common/Counters.h>
#include <torch/csrc/monitor/instrumentation.h>

#include <chrono>
#include <string_view>

namespace torch {
namespace monitor {

namespace detail {
class WaitCounterImpl : public facebook::data_preproc::WaitCounterUs {
 public:
  explicit WaitCounterImpl(std::string_view key)
      : facebook::data_preproc::WaitCounterUs(key) {}
};
static detail::WaitCounterImpl& getImpl(std::string_view key) {
  auto* impl = new detail::WaitCounterImpl(key);
  return *impl;
}
} // namespace detail

WaitCounterHandle::WaitCounterHandle(std::string_view key)
    : impl_(detail::getImpl(key)) {}

void WaitCounterHandle::start(std::chrono::steady_clock::time_point now) {
  impl_.start(now);
}

void WaitCounterHandle::stop(std::chrono::steady_clock::time_point now) {
  impl_.stop(now);
}

} // namespace monitor
} // namespace torch

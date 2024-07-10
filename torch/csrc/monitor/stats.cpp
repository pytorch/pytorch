#ifndef FBCODE_CAFFE2
#include <torch/csrc/monitor/stats.h>
#include <chrono>

namespace torch {
namespace monitor {

namespace detail {
class StatImpl {};
} // namespace detail

IntegralStat::IntegralStat(std::string_view key)
    : impl_(nullptr) {}

void IntegralStat::addValue(
    double value,
    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now()) {
  // implement
}

} // namespace monitor
} // namespace torch
#endif

#ifndef FBCODE_CAFFE2
#include <torch/csrc/monitor/stats.h>
#include <chrono>
#include <string>

namespace torch {
namespace monitor {

void registerCallback(
    const std::string& key,
    const std::function<double()>& callback) {
  // implement
}
void unregisterCallback(const std::string& key) {
  // implement
}

namespace detail {
class StatImpl {};
} // namespace detail

PeriodicAvgStat::PeriodicAvgStat(std::string_view key) : impl_(nullptr) {}

void PeriodicAvgStat::addValue(
    double value,
    std::chrono::steady_clock::time_point now) {
  // implement
}

PeriodicSumStat::PeriodicSumStat(std::string_view key) : impl_(nullptr) {}

void PeriodicSumStat::addValue(
    double value,
    std::chrono::steady_clock::time_point now) {
  // implement
}

} // namespace monitor
} // namespace torch
#endif

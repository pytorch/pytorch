#ifndef FBCODE_CAFFE2
#include <torch/csrc/monitor/stats.h>
#include <chrono>

namespace torch {
namespace monitor {

void registerCallback(std::string key, const std::function<double()>& callback) {
  // implement
}
void unregisterCallback(std::string key) {
  // implement
}

namespace detail {
class StatImpl {};
} // namespace detail

PeriodicStat::PeriodicStat(std::string_view key)
    : impl_(nullptr) {}

void PeriodicStat::addValue(
    double value,
    std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now()) {
  // implement
}

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

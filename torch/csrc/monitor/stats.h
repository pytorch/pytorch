#pragma once
#include <chrono>
#include <functional>
#include <memory>
#include <string_view>

namespace torch {
namespace monitor {

void registerCallback(
    const std::string& key,
    const std::function<double()>& callback);
void unregisterCallback(const std::string& key);

namespace detail {
class StatImpl;
} // namespace detail

class PeriodicAvgStat {
 public:
  explicit PeriodicAvgStat(std::string_view key);
  void addValue(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());

 private:
  std::shared_ptr<detail::StatImpl> impl_;
};

class PeriodicSumStat {
 public:
  explicit PeriodicSumStat(std::string_view key);
  void addValue(
      double value,
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());

 private:
  std::shared_ptr<detail::StatImpl> impl_;
};

} // namespace monitor
} // namespace torch

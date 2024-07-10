# pragma once
#include <chrono>
#include <string_view>
#include <memory>

namespace torch {
namespace monitor {

namespace detail {
class StatImpl;
} // namespace detail

class IntegralStat {
 public: 
  explicit IntegralStat(std::string_view key);
  void addValue(double value, std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());
 private:
  std::shared_ptr<detail::StatImpl> impl_;
};

} // namespace monitor
} // namespace torch

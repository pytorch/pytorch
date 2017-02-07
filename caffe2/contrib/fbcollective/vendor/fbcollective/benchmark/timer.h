#pragma once

#include <algorithm>
#include <chrono>

namespace fbcollective {
namespace benchmark {

class Timer {
 public:
  Timer() {
    start();
  }

  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  long ns() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::nanoseconds(now - start_).count();
  }

 protected:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class Distribution {
 public:
  Distribution() {
    constexpr auto capacity = 100 * 1000;
    samples_.reserve(capacity);
  }

  void clear() {
    samples_.clear();
    sorted_.clear();
  }

  size_t size() const {
    return samples_.size();
  }

  void add(long ns) {
    samples_.push_back(ns);
    sorted_.clear();
  }

  void add(const Timer& t) {
    add(t.ns());
  }

  long min() {
    return sorted()[0];
  }

  long max() {
    return sorted()[size() - 1];
  }

  long percentile(float pct) {
    return sorted()[pct * size()];
  }

  std::vector<long> sorted() {
    if (sorted_.size() != samples_.size()) {
      sorted_ = samples_;
      std::sort(sorted_.begin(), sorted_.end());
    }
    return sorted_;
  }

 protected:
  std::vector<long> samples_;
  std::vector<long> sorted_;
};

} // namespace benchmark
} // namespace fbcollective

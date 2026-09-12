// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <cmath>
#include <cstddef>
#include <deque>
#include <string>

namespace at::native::tunable {

/** http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance */

class Stats {
 public:
  Stats() {
    _n = 0UL;
    _mean = 0.0;
    _M2 = 0.0;
    _sum = 0.0;
    _min = 0.0;
    _max = 0.0;
  }

  void sample_value(const double x) {
    double delta = 0;
    _sum = _sum + x;
    if (0UL == _n) {
      _min = x;
      _max = x;
    } else {
      _min = _min < x ? _min : x;
      _max = _max > x ? _max : x;
    }
    _n = _n + 1UL;
    delta = x - _mean;
    _mean = _mean + delta / _n;
    _M2 = _M2 + delta * (x - _mean);
  }

  double variance() const {
    return _M2 / (_n - 1);
  }

  double stddev() const {
    return std::sqrt(variance());
  }

  unsigned long _n;
  double _mean;
  double _M2;
  double _sum;
  double _min;
  double _max;
};

class FixedSizeStack {
 private:
  std::deque<std::string> stack;
  const size_t max_size;

 public:
  FixedSizeStack(size_t size) : max_size(size) {}

  void push(const std::string& value) {
    if (stack.size() >= max_size) {
      stack.pop_front(); // Remove the oldest entry
    }
    stack.push_back(value); // Add new entry
  }

  auto rbegin() {
    return stack.rbegin();
  }
  auto rend() {
    return stack.rend();
  }
};

} // namespace at::native::tunable

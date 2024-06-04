// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>
#include <string>
#include <string_view>

namespace torch {
namespace monitor {

// This counter is useful to measure how much time is spent waiting on certain
// operation kind. Note that it only tracks the overall wait time rather than
// individual wait events.
// The wait time is tracked as long as there is at least one waiter.
// The implementation is thread-safe.
class WaitCounterUs {
 public:
  explicit WaitCounterUs(std::string_view key);
  ~WaitCounterUs();

  // Starts a waiter
  void start(
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());
  // Stops the waiter. Each start() call should be matched by exactly one stop()
  // call.
  void stop(
      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now());

  struct State;

 private:
  const std::string key_;
};
} // namespace monitor
} // namespace torch

// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>

namespace c10d::nccl2 {

// Process-level singleton tracking whether any NCCL2 communicator has
// experienced a watchdog timeout. Used by the debug server's health check
// endpoint to detect unhealthy ranks and trigger flight recorder dumps.
class HealthCheck {
 public:
  static HealthCheck* get() {
    // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
    static auto* instance = new HealthCheck();
    return instance;
  }

  void setTimedOut() {
    timed_out_.store(true, std::memory_order_release);
  }

  bool isTimedOut() const {
    return timed_out_.load(std::memory_order_acquire);
  }

 private:
  HealthCheck() = default;
  std::atomic<bool> timed_out_{false};
};

} // namespace c10d::nccl2

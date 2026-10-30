// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#ifdef USE_C10D_NCCL

#include <memory>
#include <string>

namespace c10d::nccl2 {

struct Heartbeat;

// Reads the monitor's environment once per process. Throws on malformed
// values, so call it where that can fail group creation, before starting the
// watchdog.
void initHeartbeatMonitor();

// Registers a watchdog thread with the process-wide heartbeat monitor for the
// lifetime of this object. If beat() is not called for
// TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC, the monitor dumps the flight recorder and,
// TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC later, terminates the process if still not
// called, like stock ProcessGroupNCCL's HeartbeatMonitor.
class HeartbeatRegistration {
 public:
  explicit HeartbeatRegistration(std::string log_prefix);
  ~HeartbeatRegistration();
  HeartbeatRegistration(const HeartbeatRegistration&) = delete;
  HeartbeatRegistration& operator=(const HeartbeatRegistration&) = delete;

  void beat();

 private:
  std::shared_ptr<Heartbeat> heartbeat_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL

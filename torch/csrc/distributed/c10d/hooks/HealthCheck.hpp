// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <c10/macros/Export.h>

namespace c10d {

// Process-wide health state reported by distributed backends. Health failures
// are sticky so the debug server can observe them after the detecting thread
// has moved on to failure handling.
class TORCH_API HealthCheck {
 public:
  static HealthCheck* get();

  void setUnhealthy(std::string backend);
  std::vector<std::string> unhealthyBackends() const;

 private:
  HealthCheck() = default;

  mutable std::mutex mutex_;
  std::set<std::string> unhealthy_backends_;
};

} // namespace c10d

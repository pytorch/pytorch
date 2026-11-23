// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/HealthCheck.hpp>

#include <nlohmann/json.hpp>

#include <utility>

#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace c10d {

HealthCheck* HealthCheck::get() {
  // NOLINTNEXTLINE(facebook-hte-InlinedStaticLocalVariableWarning)
  static auto* instance = new HealthCheck();
  return instance;
}

void HealthCheck::setUnhealthy(std::string backend) {
  std::lock_guard<std::mutex> lock(mutex_);
  unhealthy_backends_.insert(std::move(backend));
}

std::vector<std::string> HealthCheck::unhealthyBackends() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {unhealthy_backends_.begin(), unhealthy_backends_.end()};
}

namespace {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
control_plane::RegisterHandler healthCheckHandler(
    "c10d_health_check",
    [](const control_plane::Request&, control_plane::Response& res) {
      auto backends = HealthCheck::get()->unhealthyBackends();
      nlohmann::json status{
          {"healthy", backends.empty()},
          {"unhealthy_backends", std::move(backends)}};
      res.setContent(status.dump(), "application/json");
      res.setStatus(200);
    });

} // namespace
} // namespace c10d

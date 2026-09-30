// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/HealthCheck.hpp>

#include <string>
#include <utility>

#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace c10d::nccl2 {
namespace {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
control_plane::RegisterHandler nccl2HealthCheckHandler(
    "nccl2_health_check",
    [](const control_plane::Request&, control_plane::Response& res) {
      const bool timed_out = HealthCheck::get()->isTimedOut();
      std::string json =
          timed_out ? R"({"healthy": false})" : R"({"healthy": true})";
      res.setContent(std::move(json), "application/json");
      res.setStatus(200);
    });

} // namespace
} // namespace c10d::nccl2

#endif // USE_C10D_NCCL

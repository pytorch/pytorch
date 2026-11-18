// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/HealthCheckHook.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/hooks/HealthCheck.hpp>

namespace c10d {

namespace {

std::atomic<int64_t> next_hook_id{0x4845414c /* 'HEAL' */};

} // namespace

void HealthCheckHook::attach(const c10::intrusive_ptr<ProcessGroup>& pg) {
  TORCH_CHECK(pg, "HealthCheckHook: null process group");
  std::unordered_set<Backend*> attached;
  for (const auto& device : pg->getDeviceTypes()) {
    auto backend = pg->getBackend(device.type());
    if (!attached.insert(backend.get()).second ||
        !backend->supportsAbortHooks()) {
      continue;
    }
    auto hook = std::shared_ptr<HealthCheckHook>(
        new HealthCheckHook(backend->getBackendName()));
    backend->registerAbortHook(next_hook_id++, [hook]() { hook->onAbort(); });
  }
}

HealthCheckHook::HealthCheckHook(std::string backend_name)
    : backend_name_(std::move(backend_name)) {}

void HealthCheckHook::onAbort() const {
  HealthCheck::get()->setUnhealthy(backend_name_);
}

} // namespace c10d

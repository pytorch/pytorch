// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Bridges an NCCL process-group backend to a consumer that needs to track the
// backend's host communicator (today: symmetric memory's NCCLDevCommManager),
// without the backend depending on that consumer's headers. The consumer
// installs the hooks; the backend fires publishNCCLComm / retireNCCLComm on
// comm create / destroy. `comm` is an opaque host ncclComm_t, cast back on the
// consumer side, so this header stays free of nccl.h. Both entry points are
// no-ops until hooks are installed, so a build without symmetric-memory support
// -- or a comm created before the consumer is ever used -- simply does nothing.

#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

#include <functional>
#include <string>

namespace c10d {

struct NCCLCommRegistrationHooks {
  std::function<void(const std::string& group_name, void* comm, c10::Device)>
      on_register;
  std::function<void(const std::string& group_name, void* comm, c10::Device)>
      on_unregister;
};

// Installed once (typically at load time) by the consumer of the comm.
TORCH_API void setNCCLCommRegistrationHooks(NCCLCommRegistrationHooks hooks);

// Fired by comm producers; no-op if no hooks are installed.
TORCH_API void publishNCCLComm(
    const std::string& group_name,
    void* comm,
    c10::Device device);
TORCH_API void retireNCCLComm(
    const std::string& group_name,
    void* comm,
    c10::Device device);

} // namespace c10d

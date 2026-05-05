// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

namespace c10d {

// Internal extension point for c10d backends that can expose an underlying
// NCCL communicator to c10d-only integrations such as NCCL symmetric memory.
class NCCLCommProvider {
 public:
  virtual ~NCCLCommProvider() = default;
  virtual void* getNCCLCommPtr() const = 0;
};

} // namespace c10d

#pragma once

#include <chrono>
#include <memory>

#include <c10/util/Registry.h>

namespace c10d::cuda {

enum BatonStatus : int32_t {
  UNKNOWN = 0,
  RUNNING = 1,
  TIMED_OUT = 2,
  ABORTED = 3,
};

/*
Baton implements a baton that will block a the active CUDA stream
until aborted by the main process.
*/
class TORCH_API Baton {
 public:
  virtual ~Baton() = default;
  virtual void abort() = 0;
  virtual BatonStatus status() = 0;
};

std::unique_ptr<Baton> baton(std::chrono::milliseconds timeout);

TORCH_DECLARE_REGISTRY(BatonRegistry, Baton, std::chrono::milliseconds);

} // namespace c10d::cuda

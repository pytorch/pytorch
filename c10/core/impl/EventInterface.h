#pragma once

#include "c10/core/DeviceType.h"


namespace c10 {

class Stream;

// TODO: add comment (why is this here? what does it do?)
enum class EventFlag {
    PYTORCH_DEFAULT,
    BACKEND_DEFAULT,
    // CUDA flags
    CUDA_EVENT_DEFAULT,
    CUDA_EVENT_DISABLE_TIMING, // PyTorch-default for CUDA
    // HIP flags
    HIP_EVENT_DEFAULT,
    HIP_EVENT_DISABLE_TIMING, // PyTorch-default for HIP
    // FOR TESTING ONLY
    INVALID
};

namespace impl {

struct C10_API EventInterface {

  virtual DeviceType device_type() const noexcept = 0;
  virtual DeviceIndex device_index() const noexcept = 0;

  virtual void recordOnce(const Stream& stream) = 0;
  virtual void record(const Stream& stream) = 0;
  virtual void block(const Stream& stream) const = 0;
  virtual bool query() const = 0;

};

} // impl
} // c10

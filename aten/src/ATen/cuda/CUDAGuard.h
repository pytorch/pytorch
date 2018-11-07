#pragma once

#include <ATen/DeviceGuard.h>
#include <ATen/core/ArrayRef.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <vector>

namespace at { namespace cuda {

/// A variant of `DeviceGuard` that augments it with an understanding of CUDA
/// streams. This guard can not only set and reset the current CUDA device, but
/// also set and reset the current CUDA stream. It is important to note that
/// because a CUDA stream is intrinsically associated with the CUDA device to
/// which it is bound, setting the CUDA stream *also* sets the current CUDA
/// device to that of the stream.
struct CUDAGuard {
  /// Default constructor, does nothing and causes no change in the current
  /// stream or device until `set_stream` or `set_device` is called.
  CUDAGuard() = default;

  /// Sets the CUDA stream and its associated device as the current one (calls
  /// `set_stream`).
  explicit CUDAGuard(const CUDAStream& stream) {
    set_stream(stream);
  }

  /// Calls `set_device` with the given index.
  explicit CUDAGuard(int32_t device) {
    set_device(device);
  }

  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  /// Move-constructs this `CUDAGuard` from another `CUDAGuard`. The
  /// moved-from `CUDAGuard` is modified such that its destruction has no
  /// effect (does not reset the stream or device).
  CUDAGuard(CUDAGuard&& other) noexcept = default;

  /// Move-assigns this `CUDAGuard` from another `CUDAGuard`. The
  /// moved-from `CUDAGuard` is modified such that its destruction has no
  /// effect (does not reset the stream or device).
  CUDAGuard& operator=(CUDAGuard&& other) {
    device_guard_ = std::move(other.device_guard_);
    original_streams_ = std::move(other.original_streams_);
    other.original_streams_.clear();
    return *this;
  }

  /// Resets the CUDA stream on each device to the one that was active upon
  /// construction.
  ~CUDAGuard() {
    if (!original_streams_.empty()) {
      for (size_t device = 0; device < original_streams_.size(); ++device) {
        uncheckedSetCurrentCUDAStream(original_streams_[device]);
      }
    }
  }

  /// Sets the current CUDA device to the device associated with the given
  /// stream, and then sets the current stream on that device to the one given.
  void set_stream(const CUDAStream& stream) {
    set_index(stream.device_index());
    // If we haven't stored the current stream yet, store it now.
    if (original_streams_.empty()) {
      const size_t device_count = getNumGPUs();
      original_streams_.reserve(device_count);
      for (size_t device = 0; device < device_count; ++device) {
        original_streams_.push_back(getCurrentCUDAStream(device));
      }
    }
    setCurrentCUDAStream(stream);
  }

  /// Sets the CUDA device to the given one.
  /// TODO: Deprecate this name
  void set_device(int32_t device_index) {
    set_index(device_index);
  }

  /// Sets the CUDA device to the given one.
  void set_index(int32_t device_index) {
    device_guard_.set_device(at::Device(at::kCUDA, device_index));
  }

  /// Returns the CUDA streams that were active in the first call to
  /// `set_stream`. If there was no such call, the returned container is
  /// empty.
  ArrayRef<CUDAStream> original_streams() const noexcept {
    return original_streams_;
  }

  /// Returns the device that was set upon construction of the guard.
  Device original_device() const noexcept {
    return device_guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any.
  Device last_device() const noexcept {
    return device_guard_.last_device();
  }

 private:
  /// The guard for the current device.
  at::DeviceGuard device_guard_;
  /// The original streams that were active on all devices.
  std::vector<CUDAStream> original_streams_;
};

} // namespace cuda
} // namespace at

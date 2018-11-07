#pragma once

#include <ATen/DeviceGuard.h>
#include <ATen/core/ArrayRef.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/CUDAGuardImpl.h>
#include <c10/DeviceType.h>
#include <c10/detail/InlineDeviceGuard.h>

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
  /// Calls `set_device` with the given index.
  explicit CUDAGuard(DeviceIndex device) : device_guard_(device) {}

  // Copy is not allowed
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  CUDAGuard(CUDAGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  CUDAGuard& operator=(CUDAGuard&& other) = delete;

  /// Sets the CUDA device to the given one.
  /// TODO: Deprecate this name
  void set_device(DeviceIndex device_index) {
    device_guard_.set_index(device_index);
  }

  /// Sets the CUDA device to the given one.
  void set_index(DeviceIndex device_index) {
    device_guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const {
    return device_guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise the
  /// device passed during construction.
  Device current_device() const {
    return device_guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::detail::InlineDeviceGuard<detail::CUDAGuardImpl> device_guard_;
};

using MaybeCUDAGuard = c10::detail::InlineMaybeDeviceGuard<detail::CUDAGuardImpl>;

struct CUDAStreamGuard {
  /// Sets the CUDA stream and its associated device as the current one (calls
  /// `set_stream`).
  explicit CUDAStreamGuard(const CUDAStream& stream)
    : guard_(stream.device_index())
    , original_stream_(getCurrentCUDAStream(stream.device_index())) {
    setCurrentCUDAStream(stream);
  }

  /// Resets the CUDA stream on each device to the one that was active upon
  /// construction.
  ~CUDAStreamGuard() {
    uncheckedSetCurrentCUDAStream(original_stream_);
  }

  /// Returns the CUDA stream that was active in the first call to
  /// `set_stream`. If there was no such call, the returned container is
  /// empty.
  CUDAStream original_stream() const noexcept {
    return original_stream_;
  }

  Device original_device() const noexcept {
    return guard_.original_device();
  }

  Device current_device() const noexcept {
    return guard_.current_device();
  }

private:
  CUDAGuard guard_;
  CUDAStream original_stream_;
};

struct CUDAMultiStreamGuard {
  /// Calls `set_stream` on each of the streams in the list.
  /// This may be useful if you need to set different streams
  /// for different devices.
  explicit CUDAMultiStreamGuard(ArrayRef<CUDAStream> streams) {
    const size_t device_count = getNumGPUs();
    original_streams_.reserve(device_count);
    for (size_t device = 0; device < device_count; ++device) {
      original_streams_.push_back(getCurrentCUDAStream(device));
    }
    for (const auto& s : streams) {
      setCurrentCUDAStream(s);
    }
  }

  CUDAMultiStreamGuard(const CUDAGuard&) = delete;
  CUDAMultiStreamGuard& operator=(const CUDAGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  CUDAMultiStreamGuard(CUDAGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  CUDAMultiStreamGuard& operator=(CUDAGuard&& other) = delete;

  ArrayRef<CUDAStream> original_streams() const {
    return original_streams_;
  }

  /// Resets the CUDA stream on each device to the one that was active upon
  /// construction.
  ~CUDAMultiStreamGuard() {
    for (const auto& s : original_streams_) {
      uncheckedSetCurrentCUDAStream(s);
    }
  }

private:
  /// The original streams that were active on all devices.
  std::vector<CUDAStream> original_streams_;
};

} // namespace cuda
} // namespace at

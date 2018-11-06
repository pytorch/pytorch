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
  /// Default constructor.  Although no change in device occurs, the resulting
  /// CUDAGuard will still record the current device at the time of
  /// construction, and restore it when the guard goes out of scope.
  ///
  /// If you want to avoid performing the device set/get entirely (if the
  /// CUDAGuard is never actually used), you can implement this using
  /// the following idiom:
  ///
  ///     optional<CUDAGuard> g;
  ///     if (want_guard) {
  ///       g.emplace(device_index);
  ///     }
  ///
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

  /// Calls `set_stream` on each of the streams in the list.
  /// This may be useful if you need to set different streams
  /// for different devices.
  explicit CUDAGuard(ArrayRef<CUDAStream> streams) {
    for (const auto& s : streams) {
      set_stream(s);
    }
  }

  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  CUDAGuard(CUDAGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  CUDAGuard& operator=(CUDAGuard&& other) = delete;

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
  Device original_device() {
    return device_guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any.
  Device current_device() {
    return device_guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::detail::InlineDeviceGuard<detail::CUDAGuardImpl> device_guard_;
  /// The original streams that were active on all devices.
  /// TODO: Consider making stream handling another class, so we don't need
  /// to goop up the generated code with stream saving...
  std::vector<CUDAStream> original_streams_;
};

} // namespace cuda
} // namespace at

#pragma once

#include <ATen/CUDAStream.h>
#include <ATen/Context.h>
#include <ATen/optional.h>

#include <cstddef>

namespace at {

/// RAII guard that sets the current CUDA Stream for a particular CUDA device,
/// identified by its device index, and sets it back to the original stream upon
/// destruction.
///
/// The stream is always reset to the one that was active at the time of
/// construction of the guard. Even if you `set_stream` after construction, the
/// destructor will still reset the stream to the one that was active at
/// construction time.
struct CUDAStreamGuard {
  /// Default constructor, does nothing and causes no change in the current
  /// stream until `set_stream` is called.
  CUDAStreamGuard() = default;

  /// Sets the CUDA stream on the given CUDA device (calls `set_stream`).
  CUDAStreamGuard(int32_t device_index, const CUDAStream& stream) {
    set_stream(device_index, stream);
  }

  /// Resets the CUDA stream on the stored device to the one that was active at
  /// the time of construction of this object.
  ~CUDAStreamGuard() {
    if (original_stream_) {
      globalContext().uncheckedSetCurrentCUDAStreamOnDevice(
          device_index_, *original_stream_);
    }
  }

  /// Sets the CUDA stream on the CUDA device identified by the given
  /// `device_index` to the one specified by `stream`.
  void set_stream(int32_t device_index, const CUDAStream& stream) {
    device_index_ = device_index;
    // If we haven't stored the current stream yet, store it now.
    if (!original_stream_) {
      original_stream_ =
          globalContext().getCurrentCUDAStreamOnDevice(device_index);
    }
    globalContext().setCurrentCUDAStreamOnDevice(device_index, stream);
  }

  /// Returns the device index specified in the latest call to `set_stream`, or
  /// -1 if it was never called.
  int32_t device_index() const noexcept {
    return device_index_;
  }

  /// Returns the CUDA stream specified in the latest call to `set_stream`, or
  /// `nullopt` if it was never called.
  const at::optional<CUDAStream>& original_stream() const noexcept {
    return original_stream_;
  }

 private:
  /// The device index for which we are setting the stream.
  int32_t device_index_{-1};
  /// The original stream that was active on the device.
  at::optional<CUDAStream> original_stream_;
};

} // namespace at

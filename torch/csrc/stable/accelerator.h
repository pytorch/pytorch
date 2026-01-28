#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <memory>

HIDDEN_NAMESPACE_BEGIN(torch, stable, accelerator)

using DeleterFnPtr = void (*)(void*);

namespace {
inline void delete_device_guard(void* ptr) {
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_delete_device_guard(reinterpret_cast<DeviceGuardHandle>(ptr)));
}

} // namespace

// This is bigger than DeviceIndex in c10/core/Device.h but it is the type we
// can converge on in this world as DeviceIndex in libtorch is not stable.
/**
 * @brief Device index type for stable ABI.
 *
 * Minimum compatible version: PyTorch 2.9.
 */
using DeviceIndex = int32_t;

using StreamId = int64_t; // this is from c10/core/Stream.h

/**
 * @brief A stable ABI version of c10::DeviceGuard.
 *
 * RAII class that sets the current device to the specified device index
 * on construction and restores the previous device on destruction.
 *
 * Minimum compatible version: PyTorch 2.9.
 */
class DeviceGuard {
 public:
  /// \private
  explicit DeviceGuard() = delete;

  /**
   * @brief Constructs a DeviceGuard that sets the current device.
   *
   * @param device_index The device index to set as the current device.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  explicit DeviceGuard(DeviceIndex device_index)
      : guard_(nullptr, delete_device_guard) {
    DeviceGuardHandle ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_create_device_guard(device_index, &ptr));
    guard_.reset(ptr);
  }

  /**
   * @brief Changes the current device to the specified device index.
   *
   * @param device_index The new device index to set.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  void set_index(DeviceIndex device_index) {
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_device_guard_set_index(guard_.get(), device_index));
  }

 private:
  std::unique_ptr<DeviceGuardOpaque, DeleterFnPtr> guard_;
};

class Stream {
 public:
  explicit Stream() = delete;

  // Construct a stable::Stream from a StreamHandle
  // Steals ownership from the StreamHandle
  explicit Stream(StreamHandle stream)
      : stream_(stream, [](StreamHandle stream) {
          TORCH_ERROR_CODE_CHECK(aoti_torch_delete_stream(stream));
        }) {}

  StreamId id() const {
    StreamId stream_id;
    TORCH_ERROR_CODE_CHECK(aoti_torch_stream_id(stream_.get(), &stream_id));
    return stream_id;
  }

 private:
  std::shared_ptr<StreamOpaque> stream_;
};

inline Stream getCurrentStream(DeviceIndex device_index) {
  StreamHandle stream = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_stream(device_index, &stream));
  return Stream(stream);
}

/**
 * @brief Gets the current device index.
 *
 * Returns the index of the currently active device for the accelerator.
 *
 * @return The current device index.
 *
 * Minimum compatible version: PyTorch 2.9.
 */
inline DeviceIndex getCurrentDeviceIndex() {
  DeviceIndex device_index;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_device_index(&device_index));
  return device_index;
}

HIDDEN_NAMESPACE_END(torch, stable, accelerator)

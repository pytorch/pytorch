#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/TensorImpl.h>

#include <include/openreg.h>

#include <atomic>

#include "OpenRegFunctions.h"
#include "OpenRegGenerator.h"

namespace c10::openreg {

// Stands in for a real backend-specific TensorImpl subclass (analogous to
// e.g. torch_npu's NPUTensorImpl): its only purpose here is to demonstrate,
// end-to-end through Python, that fromBlobPrivateUse1() can hand back a
// tensor whose impl is genuinely not the generic c10::TensorImpl. It adds no
// behavior of its own.
class OpenRegFromBlobTensorImpl : public c10::TensorImpl {
 public:
  OpenRegFromBlobTensorImpl(
      c10::Storage&& storage,
      c10::DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type)
      : c10::TensorImpl(std::move(storage), dispatch_key, data_type) {}
};

// Counts fromBlobPrivateUse1() invocations; exposed to Python as
// torch.ops.openreg._from_blob_hook_call_count() for tests to assert on.
inline std::atomic<int64_t> g_from_blob_hook_call_count{0};

struct OPENREG_EXPORT OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  OpenRegHooksInterface() {};
  ~OpenRegHooksInterface() override = default;

  void init() const override {
    // Initialize OpenReg runtime if needed
    // This is called when PyTorch first accesses the device
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    return true;
  }

  bool isBuilt() const override {
    // This extension is compiled as part of the OpenReg test extension.
    return true;
  }

  bool isAvailable() const override {
    // Consider OpenReg available if there's at least one device reported.
    return device_count() > 0;
  }

  DeviceIndex deviceCount() const override {
    return device_count();
  }

  void setCurrentDevice(DeviceIndex device) const override {
    set_device(device);
  }

  DeviceIndex getCurrentDevice() const override {
    return current_device();
  }

  DeviceIndex exchangeDevice(DeviceIndex device) const override {
    return ExchangeDevice(device);
  }

  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override {
    // Only exchange if the requested device is valid; otherwise, no-op and return current
    auto count = device_count();
    if (device < 0 || device >= count) {
      return getCurrentDevice();
    }
    return exchangeDevice(device);
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  bool isPinnedPtr(const void* data) const override {
    orPointerAttributes attr{};
    orPointerGetAttributes(&attr, data);

    return attr.type == orMemoryTypeHost;
  }

  at::Device getDeviceFromPtr(void* data) const override {
    orPointerAttributes attr{};
    auto err = orPointerGetAttributes(&attr, data);
    if (err == orSuccess && attr.type == orMemoryTypeDevice) {
      return at::Device(at::DeviceType::PrivateUse1, static_cast<int>(attr.device));
    } else {
      TORCH_CHECK(false, "failed to get device from pointer");
    }
    return at::Device(at::DeviceType::PrivateUse1, current_device());
  }
  // LITERALINCLUDE START: OPENREG HOOK EXAMPLES
  const at::Generator& getDefaultGenerator(DeviceIndex device_index) const override {
    return getDefaultOpenRegGenerator(device_index);
  }
  // LITERALINCLUDE END: OPENREG HOOK EXAMPLES

  at::Generator getNewGenerator(DeviceIndex device_index) const override {
    return at::make_generator<OpenRegGeneratorImpl>(device_index);
  }

  // Opts into the at::from_blob() PrivateUse1 hook (see
  // ATen/detail/PrivateUse1HooksInterface.h). This lets e.g. a DLPack
  // import of an "openreg" tensor construct an OpenRegFromBlobTensorImpl
  // instead of a generic c10::TensorImpl, without torch_openreg needing to
  // fork ATen's DLConvertor.cpp the way a real out-of-tree backend
  // (e.g. torch_npu) currently has to.
  bool hasCustomFromBlob() const override {
    return true;
  }

  at::TensorBase fromBlobPrivateUse1(
      c10::DataPtr&& data_ptr,
      std::size_t size_bytes,
      at::IntArrayRef sizes,
      at::OptionalIntArrayRef strides,
      std::optional<int64_t> storage_offset,
      const at::TensorOptions& options,
      bool resizeable,
      c10::Allocator* allocator) const override {
    g_from_blob_hook_call_count++;

    c10::Storage storage{
        c10::Storage::use_byte_size_t{},
        size_bytes,
        std::move(data_ptr),
        /*allocator=*/allocator,
        /*resizable=*/resizeable};

    at::TensorBase tensor = at::detail::make_tensor_base<
        OpenRegFromBlobTensorImpl>(
        std::move(storage), options.computeDispatchKey(), options.dtype());

    auto* impl = tensor.unsafeGetTensorImpl();
    if (strides) {
      impl->set_sizes_and_strides(sizes, *strides);
    } else {
      impl->set_sizes_contiguous(sizes);
    }
    if (storage_offset) {
      impl->set_storage_offset(*storage_offset);
    }
    impl->set_requires_grad(options.requires_grad());
    return tensor;
  }
};

} // namespace c10::openreg

#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Generator.h>
#include <c10/core/Device.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
namespace at {

struct TORCH_API PrivateUse1HooksInterface {
  virtual ~PrivateUse1HooksInterface() = default;
  virtual const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDefaultGenerator`.");
  }

  virtual at::Device getDeviceFromPtr(void* data) const {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "You should register `PrivateUse1HooksInterface` for PrivateUse1 before call `getDeviceFromPtr`.");
  }

  virtual void initPrivateUse1() const {}
  virtual void resizePrivateUse1Bytes(at::Storage &storage, size_t newsize) const {
    ptrdiff_t size_bytes_i = newsize;
    TORCH_CHECK(
        !c10::overflows<int64_t>(size_bytes_i),
        "Requested storage size (",
        size_bytes_i,
        ") cannot be represented as a int64_t");
    const auto size_bytes = static_cast<int64_t>(size_bytes_i);
    void* original_data_ptr = storage.data_ptr().get();

    auto src_option =
        c10::TensorOptions().device(storage.device()).dtype(at::kByte);
    auto src_tensor = at::empty({0}, {}, src_option).set_(storage);
    src_tensor.resize_({size_bytes});

    if (original_data_ptr == src_tensor.storage().data_ptr().get()) {
      auto new_tensor = at::empty(src_tensor.sizes(), src_tensor.options());
      new_tensor.copy_(src_tensor);
      storage.set_data_ptr_noswap(
          std::move(new_tensor.storage().mutable_data_ptr()));
      storage.unsafeGetStorageImpl()->set_allocator(
          new_tensor.storage().unsafeGetStorageImpl()->allocator());
      storage.set_nbytes(new_tensor.storage().nbytes());
    }
  }
};

struct TORCH_API PrivateUse1HooksArgs {};

TORCH_API void RegisterPrivateUse1HooksInterface(at::PrivateUse1HooksInterface* hook_);

TORCH_API at::PrivateUse1HooksInterface* GetPrivateUse1HooksInterface();

TORCH_API bool isPrivateUse1HooksRegistered();

}

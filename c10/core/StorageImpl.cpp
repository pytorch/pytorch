#include <c10/core/StorageImpl.h>
#include <c10/util/flat_hash_map.h>

namespace c10 {

// The array to save function pointer for custom storageImpl create.
C10_API std::array<StorageImplCreateHelper, at::COMPILE_TIME_MAX_DEVICE_TYPES>
    StorageImplCreate;

// A allowlist of device type, currently available is PrivateUse1.
static ska::flat_hash_set<c10::DeviceType> DeviceTypeAllowList{
    DeviceType::PrivateUse1};

void SetStorageImplCreate(DeviceType t, StorageImplCreateHelper fptr) {
  // Allowlist verification.
  // Only if the devicetype is in the allowlist,
  // we allow the extension to be registered for storageImpl create.
  TORCH_CHECK(
      DeviceTypeAllowList.find(t) != DeviceTypeAllowList.end(),
      "It is only allowed to register the storageImpl create method ",
      "for PrivateUse1. ",
      "If you have related storageImpl requirements, ",
      "please expand the allowlist");
  // Register function pointer.
  int device_type = static_cast<int>(t);
  TORCH_CHECK(
      StorageImplCreate[device_type] == nullptr,
      "The StorageImplCreate function pointer for ",
      t,
      " has been registered.");
  StorageImplCreate[device_type] = fptr;
}

StorageImplCreateHelper GetStorageImplCreate(DeviceType t) {
  int device_type = static_cast<int>(t);
  return StorageImplCreate[device_type];
}

c10::intrusive_ptr<c10::StorageImpl> make_storage_impl(
    c10::StorageImpl::use_byte_size_t use_byte_size,
    c10::SymInt size_bytes,
    c10::DataPtr data_ptr,
    c10::Allocator* allocator,
    bool resizable,
    c10::optional<at::Device> device_opt) {
  // This will be non-nullptr only when there is a custom StorageImpl
  // constructor for the given device
  c10::StorageImplCreateHelper fptr = nullptr;
  if (device_opt.has_value()) {
    // We only need to check this here as this is the only case where we can
    // have a device that is not CPU (and thus for which the StorageImpl
    // constructor can be overwritten).
    fptr = c10::GetStorageImplCreate(device_opt.value().type());
  }

  if (fptr != nullptr) {
    return fptr(
        use_byte_size,
        std::move(size_bytes),
        std::move(data_ptr),
        allocator,
        resizable);
  }

  // Create a c10::StorageImpl object.
  if (data_ptr != nullptr) {
    return c10::make_intrusive<c10::StorageImpl>(
        use_byte_size,
        std::move(size_bytes),
        std::move(data_ptr),
        allocator,
        resizable);
  }
  return c10::make_intrusive<c10::StorageImpl>(
      use_byte_size, std::move(size_bytes), allocator, resizable);
}

} // namespace c10

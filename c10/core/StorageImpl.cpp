#include <c10/core/StorageImpl.h>

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

} // namespace c10

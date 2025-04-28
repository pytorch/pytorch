#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct StorageImpl;
class DataPtr;
} // namespace c10

namespace c10::impl::cow {

// Creates a Copy-on-write (COW) clone of the given storage. This will also
// convert the given storage into a COW storage if it is not COW already.
//
// Converting the storage into a COW storage will not be successful if the
// storage's DataPtr has some context (`DataPtr::get_context()`) which is not
// equal to the data pointer (`DataPtr::get()`). In this case, a nullptr is
// returned.
C10_API c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage);

// Create a COW clone of the storage, but the output storage is assigned the
// given device and allocator. Upon materialization, the data is copied from the
// original device to the new device using the given allocator.
C10_API c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage,
    Device device,
    Allocator& allocator);

// Check if a storage has a simple DataPtr with no abnormal context
C10_API bool has_simple_data_ptr(const c10::StorageImpl& storage);

// Check if a DataPtr is COW
C10_API bool is_cow_data_ptr(const c10::DataPtr& data_ptr);

// Check if a DataPtr is COW and the DataPtr points to data on the given device.
C10_API bool is_cow_data_ptr_on_device(
    const c10::DataPtr& data_ptr,
    DeviceType device_type);

// Eagerly copies a COW storage's data, turning it into a non-COW storage.
C10_API void materialize_cow_storage(StorageImpl& storage);

} // namespace c10::impl::cow

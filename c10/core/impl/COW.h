#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
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
//
// If `device_opt` is given, the output will be copied to the specified device
// when materialization occurs.
C10_API c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage,
    optional<Device> device_opt = nullopt,
    optional<Allocator*> allocator_opt = nullopt);

// Check if a storage has a simple DataPtr with no abnormal context
C10_API bool has_simple_data_ptr(const c10::StorageImpl& storage);

// Check if a DataPtr is COW. If `device_type_opt` is given, also checks if the
// DataPtr points to data on that device.
C10_API bool is_cow_data_ptr(
    const c10::DataPtr& data_ptr,
    optional<DeviceType> device_type_opt = nullopt);

// Eagerly copies a COW storage's data, turning it into a non-COW storage.
C10_API void materialize_cow_storage(StorageImpl& storage);

} // namespace c10::impl::cow

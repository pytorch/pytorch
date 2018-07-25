#include <ATen/Storage.h>
#include <ATen/Context.h>

namespace at {

// Storage::Storage(
//     at::ScalarType scalar_type,
//     ptrdiff_t size,
//     at::DataPtr data_ptr,
//     at::Allocator* allocator,
//     char flag) {
//   storage_impl_ = new StorageImpl(scalar_type, size, data_ptr, allocator, flag);
// }
// 
// Storage::Storage(
//     at::ScalarType scalar_type,
//     ptrdiff_t size,
//     at::Allocator* allocator,
//     char flag) {
//   storage_impl_ = new StorageImpl(scalar_type, size, allocator, flag);
// }

Storage::~Storage() {
  if (!storage_impl_) {
    return;
  }
  if (storage_impl_->flag & StorageImpl::REFCOUNTED) {
    if (--storage_impl_->refcount == 0) {
      if (storage_impl_->finalizer) {
        (*storage_impl_->finalizer)();
      }
      storage_impl_->finalizer = nullptr;
      storage_impl_->data_ptr.clear();
      if (storage_impl_ && --storage_impl_->weakcount == 0) {
        storage_impl_ = nullptr;
      }
    }
  }
}

} // namespace at

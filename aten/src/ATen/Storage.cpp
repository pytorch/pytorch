#include <ATen/Storage.h>
#include <ATen/Context.h>

namespace at {

Storage::~Storage() {
  if (!storage_impl_) {
    return;
  }
  if (--storage_impl_->refcount == 0) {
    if (storage_impl_->finalizer) {
      (*storage_impl_->finalizer)();
    }
    storage_impl_->finalizer = nullptr;
    storage_impl_->data_ptr.clear();
    if (storage_impl_ && --storage_impl_->weakcount == 0) {
      delete storage_impl_;
    }
  }
}

// Type& Storage::type() {
//   if (storage_impl_->data_ptr.device().is_cuda())
//     return globalContext().getType(Backend::CUDA, storage_impl_->scalar_type);
//   return globalContext().getType(Backend::CPU, storage_impl_->scalar_type);
// }

} // namespace at

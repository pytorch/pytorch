#include <ATen/core/Tensor.h>

namespace at {

inline Tensor unsafeTensorFromTH(void * th_pointer, bool retain) {
  auto tensor_impl = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(static_cast<TensorImpl*>(th_pointer));
  if (retain && tensor_impl.get() != UndefinedTensorImpl::singleton()) {
    c10::raw::intrusive_ptr::incref(tensor_impl.get());
  }
  return Tensor(std::move(tensor_impl));
}

inline Storage unsafeStorageFromTH(void * th_pointer, bool retain) {
  if (retain && th_pointer) {
    c10::raw::intrusive_ptr::incref(static_cast<StorageImpl*>(th_pointer));
  }
  return Storage(c10::intrusive_ptr<StorageImpl>::reclaim(static_cast<StorageImpl*>(th_pointer)));
}

}

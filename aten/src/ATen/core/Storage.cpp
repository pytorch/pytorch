#include <ATen/core/Storage.h>

namespace at {

Storage::Storage(
    caffe2::TypeMeta data_type,
    size_t size,
    Allocator* allocator,
    bool resizable)
    : storage_impl_(c10::make_intrusive<StorageImpl>(
          data_type,
          size,
          allocator,
          resizable)) {}

Storage::Storage(
    caffe2::TypeMeta data_type,
    at::DataPtr data_ptr,
    size_t size,
    const std::function<void(void*)>& deleter,
    bool resizable)
    : storage_impl_(c10::make_intrusive<StorageImpl>(
          data_type,
          size,
          std::move(data_ptr),
          /* allocator */ nullptr,
          resizable)) {}

} // namespace at

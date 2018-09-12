#include <ATen/core/StorageImpl.h>

namespace at {

StorageImpl::StorageImpl(
    caffe2::TypeMeta data_type,
    int64_t numel,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : data_type_(data_type),
      data_ptr_(std::move(data_ptr)),
      numel_(numel),
      resizable_(resizable),
      allocator_(allocator) {
  if (numel > 0) {
    AT_ERROR(
        data_type.id() != caffe2::TypeIdentifier::uninitialized(),
        "Constructing a storage with meta of unknown type and non-zero \
        numel");
  }
}

StorageImpl::StorageImpl(
    caffe2::TypeMeta data_type,
    int64_t numel,
    at::Allocator* allocator,
    bool resizable)
    : StorageImpl(
          data_type,
          numel,
          allocator->allocate(data_type.itemsize() * numel),
          allocator,
          resizable) {}

} // namespace at

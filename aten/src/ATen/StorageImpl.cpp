#include <ATen/StorageImpl.h>

namespace at {

StorageImpl::StorageImpl(
    at::DataType data_type,
    ptrdiff_t size,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : data_type_(data_type),
      data_ptr_(std::move(data_ptr)),
      size_(size),
      resizable_(resizable),
      allocator_(allocator) {}

StorageImpl::StorageImpl(
    at::DataType data_type,
    ptrdiff_t size,
    at::Allocator* allocator,
    bool resizable)
    : StorageImpl(
          data_type,
          size,
          allocator->allocate(
              at::elementSize(dataTypeToScalarType(data_type)) * size),
          allocator,
          resizable) {}

} // namespace at

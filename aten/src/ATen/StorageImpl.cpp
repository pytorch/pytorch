#include <ATen/StorageImpl.h>

namespace at {

StorageImpl::StorageImpl(
    at::DataType data_type,
    int64_t numel,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : data_type_(data_type),
      data_ptr_(std::move(data_ptr)),
      numel_(numel),
      resizable_(resizable),
      allocator_(allocator) {}

StorageImpl::StorageImpl(
    at::DataType data_type,
    int64_t numel,
    at::Allocator* allocator,
    bool resizable)
    : StorageImpl(
          data_type,
          numel,
          allocator->allocate(
              at::elementSize(dataTypeToScalarType(data_type)) * numel),
          allocator,
          resizable) {}

} // namespace at

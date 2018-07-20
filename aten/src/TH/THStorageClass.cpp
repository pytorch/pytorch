#include "THStorageClass.hpp"

THStorage::THStorage(
    at::ScalarType scalar_type,
    int64_t size,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : scalar_type(scalar_type),
      data_ptr(std::move(data_ptr)),
      size(size),
      refcount(1),
      weakcount(1), // from the strong reference
      resizable_(resizable),
      allocator(allocator),
      finalizer(nullptr) {}

THStorage::THStorage(
    at::ScalarType scalar_type,
    int64_t size,
    at::Allocator* allocator,
    bool resizable)
    : THStorage(
          scalar_type,
          size,
          allocator->allocate(at::elementSize(scalar_type) * size),
          allocator,
          resizable) {}

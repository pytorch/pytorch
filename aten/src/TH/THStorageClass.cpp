#include "THStorageClass.hpp"

THStorage::THStorage(
    at::ScalarType scalar_type,
    ptrdiff_t size,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    char flag)
    : scalar_type(scalar_type),
      data_ptr(std::move(data_ptr)),
      size(size),
      refcount(1),
      weakcount(1), // from the strong reference
      flag(flag),
      allocator(allocator),
      finalizer(nullptr) {}

THStorage::THStorage(
    at::ScalarType scalar_type,
    ptrdiff_t size,
    at::Allocator* allocator,
    char flag)
    : THStorage(
          scalar_type,
          size,
          allocator->allocate(at::elementSize(scalar_type) * size),
          allocator,
          flag) {}

#include "THStorageClass.hpp"

THStorage::THStorage(
    at::ScalarType scalar_type,
    ptrdiff_t size,
    at::Allocator* allocator,
    char flag)
    : scalar_type(scalar_type),
      data_ptr(allocator->allocate(at::elementSize(scalar_type) * size)),
      size(size),
      refcount(1),
      weakcount(1), // from the strong reference
      allocator(allocator),
      finalizer(nullptr) {}

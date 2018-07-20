#include <ATen/Storage.h>
#include <ATen/Type.h>
#include "Context.h"

namespace at {

Storage::Storage(
    at::Backend backend,
    at::ScalarType scalar_type,
    int64_t size,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : backend(backend),
      scalar_type(scalar_type),
      data_ptr(std::move(data_ptr)),
      size_(size),
      refcount(1),
      weakcount(1), // from the strong reference
      resizable_(resizable),
      allocator(allocator),
      finalizer(nullptr) {}

Storage::Storage(
    at::Backend backend,
    at::ScalarType scalar_type,
    int64_t size,
    at::Allocator* allocator,
    bool resizable)
    : Storage(
          backend,
          scalar_type,
          size,
          allocator->allocate(at::elementSize(scalar_type) * size),
          allocator,
          resizable) {}

Type& Storage::type() const {
  return at::globalContext().getType(backend, scalar_type);
}

} // namespace at

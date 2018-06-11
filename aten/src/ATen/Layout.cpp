#include <ATen/Layout.h>

#include <ATen/ScalarType.h>
#include <ATen/Type.h>

namespace at {
Layout layout_from_type(const Type& type) {
  return layout_from_backend(type.backend());
}

Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
      return Layout::Sparse;
    default:
      return Layout::Strided;
  }
}
} // namespace at

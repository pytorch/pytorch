#include <ATen/core/DeprecatedTypeProperties.h>

#include <ATen/core/UnsafeFromTH.h>

namespace at {

Tensor DeprecatedTypeProperties::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  return at::unsafeTensorFromTH(th_pointer, retain);
}

Storage DeprecatedTypeProperties::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  return at::unsafeStorageFromTH(th_pointer, retain);
}

Tensor DeprecatedTypeProperties::copy(const Tensor & src, bool non_blocking, std::optional<Device> to_device) const {
  if (to_device) {
    return src.to(src.options().dtype(scalarType()).device(to_device), non_blocking, /*copy=*/true);
  }
  return src.to(src.options().dtype(scalarType()), non_blocking, /*copy=*/true);
}

} // namespace at

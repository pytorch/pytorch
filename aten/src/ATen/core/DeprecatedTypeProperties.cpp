#include <ATen/core/DeprecatedTypeProperties.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/Tensor.h>

namespace at {

Tensor DeprecatedTypeProperties::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  return getDispatchType().unsafeTensorFromTH(th_pointer, retain);
}

Storage DeprecatedTypeProperties::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  return getDispatchType().unsafeStorageFromTH(th_pointer, retain);
}

Tensor DeprecatedTypeProperties::copy(const Tensor & src, bool non_blocking, c10::optional<Device> to_device) const {
  if (to_device) {
    return src.to(src.options().dtype(scalarType()).device(to_device), non_blocking, /*copy=*/true);
  }
  return src.to(src.options().dtype(scalarType()), non_blocking, /*copy=*/true);
}

Type & DeprecatedTypeProperties::getDispatchType() const {
  return globalLegacyTypeDispatch().getType(backend_, scalar_type_, is_variable_);
}

} // namespace at

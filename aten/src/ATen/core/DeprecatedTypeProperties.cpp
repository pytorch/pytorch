#include <ATen/core/DeprecatedTypeProperties.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/Type.h>

namespace at {

Tensor DeprecatedTypeProperties::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  return getDispatchType().unsafeTensorFromTH(th_pointer, retain);
}

Tensor DeprecatedTypeProperties::copy(const Tensor & src, bool non_blocking, c10::optional<Device> to_device) const {
  return getDispatchType().copy(src, non_blocking, to_device);
}

std::unique_ptr<Generator> DeprecatedTypeProperties::generator() const {
  return getDispatchType().generator();
}

Type & DeprecatedTypeProperties::getDispatchType() const {
  return globalLegacyTypeDispatch().getType(backend_, scalar_type_, is_variable_);
}

} // namespace at

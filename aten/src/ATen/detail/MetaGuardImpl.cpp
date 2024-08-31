#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>

namespace at::detail {

C10_REGISTER_GUARD_IMPL(Meta, c10::impl::NoOpDeviceGuardImpl<DeviceType::Meta>);

} // namespace at::detail

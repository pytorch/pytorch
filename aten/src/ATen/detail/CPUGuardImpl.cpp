#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace at::detail {

C10_REGISTER_GUARD_IMPL(CPU, c10::impl::NoOpDeviceGuardImpl<DeviceType::CPU>)

} // namespace at::detail

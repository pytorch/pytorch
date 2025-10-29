#include <c10/core/Device.h>
#include <c10/macros/Macros.h>
#include <cstdint>

namespace at::xpu {
namespace detail {
void init_p2p_access_cache(int64_t num_devices);
} // namespace detail

TORCH_XPU_API bool get_p2p_access(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access);

} // namespace at::xpu

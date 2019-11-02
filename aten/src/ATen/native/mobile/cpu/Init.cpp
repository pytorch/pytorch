#include <ATen/native/mobile/cpu/Engine.h>
#include <xnnpack.h>

namespace at {
namespace native {
namespace mobile {
namespace cpu {

bool initialize()
{
    const xnn_status status = xnn_initialize();
    return xnn_status_success == status;
}

} // namespace cpu
} // namespace mobile
} // namespace native
} // namespace at

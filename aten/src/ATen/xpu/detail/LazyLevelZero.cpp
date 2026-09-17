#include <ATen/xpu/detail/LazyLevelZero.h>

#include <ATen/xpu/level_zero_stub/ATenLevelZero.h>
#include <stdexcept>

namespace at::xpu::detail {

LevelZero lazyLevelZero = {
// Intel level zero is not defaultly available on Windows.
#define _REFERENCE_MEMBER(name) c10::xpu::DriverAPI::get()->name##_,
    C10_LIBXPU_DRIVER_API_REQUIRED(_REFERENCE_MEMBER)
#undef _REFERENCE_MEMBER
};

} // namespace at::xpu::detail

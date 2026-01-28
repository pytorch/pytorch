#pragma once
#include <ATen/detail/XPUHooksInterface.h>
namespace at::xpu {
// Forward-declares at::xpu::LevelZero
struct LevelZero;

namespace detail {
extern LevelZero lazyLevelZero;
} // namespace detail

} // namespace at::xpu

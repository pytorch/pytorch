#pragma once
#include <ATen/detail/XPUHooksInterface.h>
namespace at::xpu {
// Forward-declares at::xpu::LevelZero
struct LevelZero;

namespace detail {
// extern declaration defines no storage; clang-tidy 21 false positive
// NOLINTNEXTLINE(bugprone-dynamic-static-initializers)
extern LevelZero lazyLevelZero;
} // namespace detail

} // namespace at::xpu

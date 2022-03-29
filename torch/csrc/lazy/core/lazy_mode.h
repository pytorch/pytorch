#pragma once

#include <atomic>
#include <c10/core/Device.h>
#include <c10/macros/Export.h>

namespace torch {
namespace lazy {

TORCH_API std::atomic<bool>& in_lazy_mode();
TORCH_API void LazyModeEnter(c10::Device device);
TORCH_API void LazyModeExit(c10::Device device);

} // namespace lazy
} // namespace torch

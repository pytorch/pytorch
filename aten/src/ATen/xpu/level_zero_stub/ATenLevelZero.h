#pragma once

#include <c10/xpu/driver_api.h>

namespace at::xpu {

// NOTE [ USE OF Level zero API ]
//
// XPU ATen does not directly link to Intel level_zero because it
// require libze_loader to be installed. Following the design of PyTorch,
// we want our GPU build to work on CPU
// machines as long as XPU is not initialized.
//
// Normal XPU code in torch uses the sycl runtime libraries which can be
// installed even if the driver is not installed, but sometimes we specifically
// need to use the driver API (e.g., to load JIT compiled code).
// To accomplish this, we lazily link the level_zero_stub which provides a
// struct at::xpu::LevelZero that contains function pointers to all of the apis
// we need.
//
// IT IS AN ERROR TO TRY TO CALL ANY ze* FUNCTION DIRECTLY.
// INSTEAD USE, e.g.
//   detail::getXPUHooks().level_zero().zeModuleCreate(...)
// or
//   globalContext().getLevelZero().zeModuleCreate(...)
//
// If a function is missing add it to the list in
// ATen/xpu/level_zero_stub/ATenLevelZero.h and edit
// ATen/xpu/detail/LazyLevelZero.cpp accordingly (e.g., via one of the stub
// macros).

extern "C" typedef struct LevelZero {
// Intel level zero is not defaultly available on Windows.
#define CREATE_MEMBER(name) decltype(&name) name;
  C10_LIBXPU_DRIVER_API_REQUIRED(CREATE_MEMBER)
#undef CREATE_MEMBER
} LevelZero;

} // namespace at::xpu

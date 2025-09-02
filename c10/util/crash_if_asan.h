#pragma once

#include <c10/macros/Macros.h>

namespace c10 {
C10_API int crash_if_asan(int arg);
C10_API int crash_if_ubsan();
} // namespace c10

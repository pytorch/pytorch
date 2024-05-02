#pragma once

#include <mkl_types.h>

namespace at::native {

  // Since size of MKL_LONG varies on different platforms (linux 64 bit, windows
  // 32 bit), we need to programmatically calculate the max.
  constexpr int64_t MKL_LONG_MAX = ((1LL << (sizeof(MKL_LONG) * 8 - 2)) - 1) * 2 + 1;

} // namespace at::native

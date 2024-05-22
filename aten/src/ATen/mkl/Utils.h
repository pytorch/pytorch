#pragma once

#include <mkl_types.h>

static inline MKL_INT mkl_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<MKL_INT>(value);
  TORCH_CHECK(
      static_cast<int64_t>(result) == value,
      "mkl_int_cast: The value of ",
      varname,
      "(",
      (long long)value,
      ") is too large to fit into a MKL_INT (",
      sizeof(MKL_INT),
      " bytes)");
  return result;
}

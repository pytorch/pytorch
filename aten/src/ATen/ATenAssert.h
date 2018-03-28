#pragma once

#include <ATen/Error.h>

#define AT_ASSERT(cond, ...) \
  if (!(cond)) {             \
    AT_ERROR(__VA_ARGS__);   \
  }

#pragma once
#include <c10/util/Exception.h>

namespace at {

enum class padding_mode {
  reflect,
  replicate,
  circular,
  constant,
};

} // namespace at

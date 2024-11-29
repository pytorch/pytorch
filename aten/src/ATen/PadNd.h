#pragma once
#include <c10/util/Exception.h>
#include <c10/util/string_view.h>

namespace at {

enum class padding_mode {
  reflect,
  replicate,
  circular,
  constant,
};

} // namespace at

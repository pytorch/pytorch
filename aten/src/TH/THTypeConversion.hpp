#pragma once

#include <ATen/Half.h>
#include "THHalf.h"

// Type traits to convert types to TH-specific types. Used primarily to
// convert at::Half to TH's half type. This makes the conversion explicit.
// FIXME: we should just use the same type

namespace th {

template <typename T>
struct FromTypeConversion {
  using type = T;
};

template <>
struct FromTypeConversion<THHalf> {
  using type = at::Half;
};

template <typename T>
using from_type = typename FromTypeConversion<T>::type;
}

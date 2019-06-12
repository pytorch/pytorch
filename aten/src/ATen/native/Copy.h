#pragma once

#include "ATen/ATen.h"

namespace at {
namespace native {

// Note [Implicit conversion between signed and unsigned]
// C and C++ have a lovely set of implicit conversion rules, where casting
// signed integral values to unsigned integral values is always valid
// (it basically treats the value as if using modulo arithmetic), however
// converting negative floating point values to unsigned integral types
// is UB! This means that: (double)-1 -> (int64_t)-1 -> (uint8_t)255 is
// guaranteed to look like this, but we have (double)-1 -> (uint8_t)<ANYTHING>
// because it's UB. This also makes UBSan really angry.
//
// I think those rules are stupid and we really shouldn't conform to them.
// The structs below ensure that for all unsigned types we use (currently
// only uint8_t), we will do an intermediate convertion via int64_t,
// to ensure that any negative values are wrapped around correctly.
//
// Note that conversions from doubles to signed integral types that can't
// represent a particular value after truncating the fracitonal part are UB as well,
// but fixing them is not as simple as adding an int64_t intermediate, beacuse the
// int64_t -> <smaller signed type> conversion is UB for those large values anyway.
// I guess in that case we just have to live with that, but it's definitely less
// surprising than the thing above.
//
// For the curious:
//   https://en.cppreference.com/w/cpp/language/implicit_conversion
//   The relevant paragraph is "Floating-integral conversions".

template <typename T>
struct inter_copy_type {
  using type = T;
};

template <>
struct inter_copy_type<uint8_t> {
  using type = int64_t;
};

template <typename T>
using inter_copy_type_t = typename inter_copy_type<T>::type;

} // namespace native
} // namespace at

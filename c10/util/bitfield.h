#pragma once

#include <c10/macros/Macros.h>

#include <type_traits>

namespace c10 {

// getBitfield/setBitfield require pos, len in [0, bit-width of T); out-of-range
// values shift by >= the width and are undefined.
template <typename T>
requires std::is_unsigned_v<T> struct Bitfield {
  C10_HOST_DEVICE static inline T getBitfield(T val, int pos, int len) {
    T m = (static_cast<T>(1) << len) - 1;
    return (val >> pos) & m;
  }

  C10_HOST_DEVICE static inline T setBitfield(
      T val,
      T toInsert,
      int pos,
      int len) {
    T m = (static_cast<T>(1) << len) - 1;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;
    return (val & ~m) | toInsert;
  }
};

} // namespace c10

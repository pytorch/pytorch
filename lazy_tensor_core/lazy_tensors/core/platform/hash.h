#pragma once

#include <functional>

#include "lazy_tensors/types.h"

namespace lazy_tensors {

extern uint64 Hash64(const char* data, size_t n, uint64 seed);

inline uint64 Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64 Hash64Combine(uint64 a, uint64 b) {
  return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
}

// Hash functor suitable for use with power-of-two sized hashtables.  Use
// instead of std::hash<T>.
//
// In particular, tensorflow::hash is not the identity function for pointers.
// This is important for power-of-two sized hashtables like FlatMap and FlatSet,
// because otherwise they waste the majority of their hash buckets.
//
// The second type argument is only used for SFNIAE below.
template <typename T, typename = void>
struct hash {
  size_t operator()(const T& t) const { return std::hash<T>()(t); }
};

template <typename T>
struct hash<T, typename std::enable_if<std::is_enum<T>::value>::type> {
  size_t operator()(T value) const {
    // This works around a defect in the std::hash C++ spec that isn't fixed in
    // (at least) gcc 4.8.4:
    // http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2148
    //
    // We should be able to remove this and use the default
    // tensorflow::hash<EnumTy>() once we stop building with GCC versions old
    // enough to not have this defect fixed.
    return std::hash<uint64>()(static_cast<uint64>(value));
  }
};

}  // namespace lazy_tensors

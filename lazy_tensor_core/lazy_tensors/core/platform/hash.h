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

}  // namespace lazy_tensors

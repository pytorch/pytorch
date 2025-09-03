#pragma once
// Set of global constants that could be shareable between CPU and Metal code

#ifdef __METAL__
#include <metal_array>
#define C10_METAL_CONSTEXPR constant constexpr
#else
#include <array>
#define C10_METAL_CONSTEXPR constexpr
#endif

#define C10_METAL_ALL_TYPES_FUNCTOR(_) \
  _(Byte, 0)                           \
  _(Char, 1)                           \
  _(Short, 2)                          \
  _(Int, 3)                            \
  _(Long, 4)                           \
  _(Half, 5)                           \
  _(Float, 6)                          \
  _(ComplexHalf, 8)                    \
  _(ComplexFloat, 9)                   \
  _(Bool, 11)                          \
  _(BFloat16, 15)

namespace c10 {
namespace metal {
C10_METAL_CONSTEXPR unsigned max_ndim = 16;
C10_METAL_CONSTEXPR unsigned simdgroup_size = 32;

#ifdef __METAL__
template <typename T, unsigned N>
using array = ::metal::array<T, N>;
#else
template <typename T, unsigned N>
using array = std::array<T, N>;
#endif

enum class ScalarType {
#define _DEFINE_ENUM_VAL_(_v, _n) _v = _n,
  C10_METAL_ALL_TYPES_FUNCTOR(_DEFINE_ENUM_VAL_)
#undef _DEFINE_ENUM_VAL_
};

} // namespace metal
} // namespace c10

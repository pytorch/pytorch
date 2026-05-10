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
  _(BFloat16, 15)                      \
  _(UInt16, 27)                        \
  _(UInt32, 28)                        \
  _(UInt64, 29)

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

// Integer ceiling division: ceil(a / b). Usable from both host code and
// Metal shaders (where the overload is selected by ADL via `using namespace
// c10::metal;` in shader sources).
template <typename T>
inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// Round `a` up to the next multiple of `b`: ceil(a / b) * b.
template <typename T>
inline T round_up(T a, T b) {
  return ceil_div(a, b) * b;
}

enum class ScalarType {
#define _DEFINE_ENUM_VAL_(_v, _n) _v = _n,
  C10_METAL_ALL_TYPES_FUNCTOR(_DEFINE_ENUM_VAL_)
#undef _DEFINE_ENUM_VAL_
};

} // namespace metal
} // namespace c10

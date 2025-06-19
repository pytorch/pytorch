#pragma once
// Set of global constants that could be shareable between CPU and Metal code

#ifdef __METAL__
#define C10_METAL_CONSTEXPR constant constexpr
#else
#define C10_METAL_CONSTEXPR constexpr
#endif

#if !defined(__METAL__) || __METAL_VERSION__ >= 310
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
#else
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
  _(Bool, 11)
#endif

namespace c10 {
namespace metal {
C10_METAL_CONSTEXPR unsigned max_ndim = 16;

enum class ScalarType {
#define _DEFINE_ENUM_VAL_(_v, _n) _v = _n,
  C10_METAL_ALL_TYPES_FUNCTOR(_DEFINE_ENUM_VAL_)
#undef _DEFINE_ENUM_VAL_
};

} // namespace metal
} // namespace c10

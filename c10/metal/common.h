#pragma once
// Set of global constants that could be shareable between CPU and Metal code

#ifdef __METAL__
#define C10_METAL_CONSTEXPR constant constexpr
#else
#define C10_METAL_CONSTEXPR constexpr
#endif

namespace c10 {
namespace metal {
C10_METAL_CONSTEXPR unsigned max_ndim = 16;

enum class ScalarType {
    Byte = 0,
    Char = 1,
    Short = 2,
    Int = 3,
    Long = 4,
    Half = 5,
    Float = 6,
    Bool = 11,
#if !defined(__METAL__) || __METAL_VERSION__ >= 310
    BFloat16 = 15,
#endif
};

} // namespace metal
} // namespace c10

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
} // namespace metal
} // namespace c10

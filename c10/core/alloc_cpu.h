#pragma once

#include <c10/macros/Macros.h>

#include <cstddef>

namespace c10 {

#ifdef C10_MOBILE
// Use 16-byte alignment on mobile
// - ARM NEON AArch32 and AArch64
// - x86[-64] < AVX
constexpr std::size_t gAlignment = 16;
#else
// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr std::size_t gAlignment = 64;
#endif

C10_API void* alloc_cpu(std::size_t nbytes);

} // namespace c10

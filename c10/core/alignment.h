#pragma once

#include <cstddef>

namespace c10 {

#ifdef C10_MOBILE
// Use 16-byte alignment on mobile
// - ARM NEON AArch32 and AArch64
// - x86[-64] < AVX
constexpr size_t gAlignment = 16;
#else
// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;
#endif

constexpr size_t gPagesize = 4096;
// since the default thp pagesize is 2MB, enable thp only
// for buffers of size 2MB or larger to avoid memory bloating
constexpr size_t gAlloc_threshold_thp = 2 * 1024 * 1024;
} // namespace c10

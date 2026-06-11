#pragma once

#include <torch/headeronly/macros/Macros.h>

#include <cstddef>
#include <new>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

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
constexpr size_t gAlloc_threshold_thp = static_cast<size_t>(2) * 1024 * 1024;

// Cache line size used to avoid false sharing between threads. Falls back to 64
// bytes if C++17 feature is unavailable.
#ifdef __cpp_lib_hardware_interference_size
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

HIDDEN_NAMESPACE_END(torch, headeronly)

namespace c10 {
using torch::headeronly::gAlignment;
using torch::headeronly::gAlloc_threshold_thp;
using torch::headeronly::gPagesize;
using torch::headeronly::hardware_destructive_interference_size;
} // namespace c10

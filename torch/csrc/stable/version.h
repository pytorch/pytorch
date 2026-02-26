#pragma once

#include <torch/headeronly/version.h>

// Stable ABI Version Targeting
//
// This header provides version targeting capabilities for the PyTorch Stable
// ABI. Users can define TORCH_TARGET_VERSION to target a specific stable ABI
// version instead of using the current TORCH_ABI_VERSION of libtorch at
// compile time.
//
// Usage:
//   Default behavior (uses current ABI version):
//     #include <torch/csrc/stable/library.h>
//
//   Target a specific stable version (major.minor) (e.g. PyTorch 2.9):
//   (1) Pass a compiler flag -DTORCH_TARGET_VERSION=0x0209000000000000
//   (2) Alternatively, define TORCH_TARGET_VERSION in the source code before
//   including any header files:
//     #define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 9) << 48))
//     #include <torch/csrc/stable/library.h>

#ifdef TORCH_TARGET_VERSION
#define TORCH_FEATURE_VERSION TORCH_TARGET_VERSION
#else
#define TORCH_FEATURE_VERSION TORCH_ABI_VERSION
#endif

#define TORCH_VERSION_2_10_0 (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
#define TORCH_VERSION_2_11_0 (((0ULL + 2) << 56) | ((0ULL + 11) << 48))
#define TORCH_VERSION_2_12_0 (((0ULL + 2) << 56) | ((0ULL + 12) << 48))

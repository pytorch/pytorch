#pragma once

#include <torch/headeronly/version.h>

// Stable ABI Version Targeting
//
// This header provides version targeting capabilities for the PyTorch Stable
// ABI. Users can define TORCH_TARGET_VERSION to target a specific stable ABI
// version instead of using the current TORCH_ABI_VERSION.
//
// Usage:
//   Default behavior (uses current ABI version):
//     #include <torch/csrc/stable/library.h>
//
//   Target a specific stable version (major.minor) (e.g. PyTorch 2.9):
//     #define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 9) << 48))
//     #include <torch/csrc/stable/library.h>

#ifdef TORCH_TARGET_VERSION
#define TORCH_FEATURE_VERSION TORCH_TARGET_VERSION
#else
#define TORCH_FEATURE_VERSION TORCH_ABI_VERSION
#endif

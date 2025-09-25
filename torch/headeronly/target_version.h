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
//   Target a specific stable version (major.minor):
//     #define TORCH_TARGET_VERSION ((uint64_t)2 << 56 | (uint64_t)9 << 48)
//     Target PyTorch 2.9 #include <torch/csrc/stable/library.h>

#ifdef TORCH_TARGET_VERSION
#define TORCH_FEATURE_VERSION TORCH_TARGET_VERSION
#else
#define TORCH_FEATURE_VERSION TORCH_ABI_VERSION
#endif

#pragma once

/// Indicates the major version of LibTorch.
#define TORCH_VERSION_MAJOR 2

/// Indicates the minor version of LibTorch.
#define TORCH_VERSION_MINOR 9

/// Indicates the patch version of LibTorch.
#define TORCH_VERSION_PATCH 0

/// Indicates the ABI version tag of LibTorch.
#define TORCH_VERSION_ABI_TAG 0

/// Indicates the version of LibTorch as a string literal.
#define TORCH_VERSION \
  "2.9.0"

/// Indicates the ABI version of LibTorch as a single uint64.
/// [ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ][ byte ]
/// [ MAJ  ][ MIN  ][ PATCH][                              ABI TAG ]
#define TORCH_ABI_VERSION \
  (uint64_t)TORCH_VERSION_MAJOR << 56 | \
  (uint64_t)TORCH_VERSION_MINOR << 48 | \
  (uint64_t)TORCH_VERSION_PATCH << 40 | \
  TORCH_VERSION_ABI_TAG << 0

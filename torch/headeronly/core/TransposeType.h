#pragma once

#include <torch/headeronly/macros/Macros.h>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// Used as an interface between the different BLAS-like libraries.
enum class TransposeType {
  NoTranspose,
  Transpose,
  ConjTranspose,
};

HIDDEN_NAMESPACE_END(torch, headeronly)

namespace at::native {
using torch::headeronly::TransposeType;
} // namespace at::native

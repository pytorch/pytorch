#pragma once

#include <torch/headeronly/macros/Macros.h>

// Keep the definition in at::native so existing forward declarations and
// callers continue to refer to the same type.
namespace at::native {

// Used as an interface between the different BLAS-like libraries.
enum class TransposeType {
  NoTranspose,
  Transpose,
  ConjTranspose,
};

} // namespace at::native

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using at::native::TransposeType;
HIDDEN_NAMESPACE_END(torch, headeronly)

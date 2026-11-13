#pragma once
#include <c10/util/Exception.h>
#include <torch/headeronly/core/TransposeType.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-default")

namespace at::native {

// Transforms TransposeType into the BLAS / LAPACK format
[[maybe_unused]] static inline char to_blas(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return 'T';
    case TransposeType::NoTranspose: return 'N';
    case TransposeType::ConjTranspose: return 'C';
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

}  // namespace at::native

C10_DIAGNOSTIC_POP()

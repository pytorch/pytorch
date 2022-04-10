#pragma once
#include <c10/util/Exception.h>

namespace at {
namespace native {

// Used as an interface between the different BLAS-like libraries
enum class TransposeType {
  NoTranspose,
  Transpose,
  ConjTranspose,
};

// Transforms TransposeType into the BLAS / LAPACK format
static char to_blas(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose: return 'T';
    case TransposeType::NoTranspose: return 'N';
    case TransposeType::ConjTranspose: return 'C';
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

}}  // at::native

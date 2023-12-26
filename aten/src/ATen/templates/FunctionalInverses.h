#pragma once

// ${generated_comment}

#include <ATen/Tensor.h>

namespace at {
namespace functionalization {

enum class InverseReturnMode {
  /// Specifies that functional inverses should always return a view.
  AlwaysView,
  /// Specifies that functional inverses should always return a non-view / copy.
  NeverView,
  /// Specifies that functional inverses should return a view unless a (copying) scatter
  /// inverse exists, in which case that will be used instead.
  /// This avoids as_strided() calls that can be difficult for subclasses to handle.
  ViewOrScatterInverse,
};

struct FunctionalInverses {

${view_inverse_declarations}

};
}
}

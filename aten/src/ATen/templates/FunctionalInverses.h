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

// NB: These are not generated! They're manually implemented in the template.
// TODO: Change codegen to generate these. See the following link:
// https://github.com/pytorch/pytorch/blob/main/torchgen/model.py#L2583-L2585
static at::Tensor chunk_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int64_t mutated_view_idx, int chunks, int dim);
static at::Tensor narrow_inverse(const at::Tensor & base, const at::Tensor & mutated_view, InverseReturnMode inverse_return_mode, int dim, c10::SymInt start, c10::SymInt length);

};
}
}

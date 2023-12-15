#pragma once

// ${generated_comment}

#include <ATen/Tensor.h>

namespace at {
namespace functionalization {

struct FunctionalInverses {

${view_inverse_declarations}

// NB: This is not generated! It's manually implemented in the template.
static at::Tensor narrow_copy_inverse(const at::Tensor & base, const at::Tensor & mutated_view, bool reapply_views, bool called_by_functionalization, int dim, c10::SymInt start, c10::SymInt length);

};
}
}

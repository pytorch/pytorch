#pragma once

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class Expr;

// Apply loop-invariant code hoisting to serial WelfordOps. For
// example, when the innermost loop looks like:
//
// for () {
//   welfordCombine(...);
/// }
//
// The count input should be invariant when the loop is not a
// reduction loop, and then this can be transformed as:
//
// After:
// nvfuser_index_t new_count = outN()[0] + 1;
// float reciprocal = 1 / new_count;
// for () {
//   welfordVectorized(..., new_count, reciprocal);
// }
//
// Here, welfordVectorized does not need to compute the division. This
// transformation can be applied when the innermost loop is a
// non-reduction domain and there's no predicate depending on the loop
// index of the innermost loop. A common case is when the read of a
// fusion input is vectorized and that input is fed to an outer
// welford reduction. In this case, the innermost domain is a
// non-reduction domain and is vectorized, so the prediacte should not
// have any dependency with the loop index, which enables the code
// moition as the above.
std::vector<Expr*> vectorizeWelford(const std::vector<Expr*>& exprs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

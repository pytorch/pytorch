#pragma once

#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <deque>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace analysis {

// A simple class containing the start and end of a range in a single dimension.
struct TORCH_API Bound {
  const Expr* start{nullptr};
  const Expr* end{nullptr};

  // This stores whether or not the start and end of this Bound have previously
  // been swapped. This occurs when the bound is in a loop with a negative
  // stride.
  bool swapped{false};

  Bound() = default;
  Bound(const Expr* s, const Expr* e) : start(s), end(e) {}

  void print() const {
    std::cout << "(" << *start << ", " << *end << ")";
  }

  bool equals(const Bound& other) const {
    return exprEquals(start, other.start) && exprEquals(end, other.end);
  }

  bool operator==(const Bound& other) const {
    return exprEquals(start, other.start) && exprEquals(end, other.end);
  }

  void swap() {
    std::swap(start, end);
    swapped = !swapped;
  }
};

struct BoundHash {
  size_t operator()(const Bound& b) const {
    return std::hash<const Expr*>()(b.start) ^ std::hash<const Expr*>()(b.end);
  }
};

// The type of overlap found. Each condition is true only if none of the
// previous conditions hold.
//     ContainedOrEqual: All elements in the Bound A are in the Bound B (this
//                       includes the case where the bounds are equal).
//     Contains: All elements in the Bound B are in the Bound B.
//     PartialOverlap: Any elements in the Bound B are in the Bound A.
//     NoOverlap: No elements in the Bound A are in the bound B.
enum OverlapKind { ContainedOrEqual, Contains, PartialOverlap, NoOverlap };

// Returns the kind of overlap between Bound A and Bound A in a single
// dimension.
OverlapKind TORCH_API boundOverlap(Bound A, Bound B);

// A multi dimensional bound representing the bound of a set of indices.
using IndexBounds = std::vector<Bound>;

// Returns true if two IndexBounds are equivalent.
bool TORCH_API indexBoundsEquals(const IndexBounds& A, const IndexBounds& B);

// Flattens a multi dimensional bound to a single dimension. The IndexBounds "a"
// *must* encapsulate the entire range of the buffer.
Bound TORCH_API flattenBounds(const IndexBounds& a);

// Determines the kind of overlap in X dimensions.
OverlapKind TORCH_API overlaps(const IndexBounds& a, const IndexBounds& b);

// Returns the Bound slices created by subtracing bound B from bound A.
// Multiple Bounds can be returned in the case where B slices A into two
// distinct regions with no overlap.
//
// Note: this doesn't use IndexBounds because the Bounds returned do not
// represent multiple different dimensions.
std::vector<Bound> TORCH_API subtractBound(Bound a, Bound b);
std::vector<Bound> TORCH_API
subtractBound(Bound a, Bound b, OverlapKind overlap);

// Returns the bound slices created by subtracting the IndexBounds B from A.
std::vector<IndexBounds> TORCH_API subtractIndicesBounds(
    const IndexBounds& A,
    const IndexBounds& B,
    OverlapKind overlap);
std::vector<IndexBounds> TORCH_API
subtractIndicesBounds(const IndexBounds& A, const IndexBounds& B);

} // namespace analysis
} // namespace tensorexpr
} // namespace jit
} // namespace torch

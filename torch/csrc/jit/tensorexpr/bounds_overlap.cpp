#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>

namespace torch {
namespace jit {
namespace tensorexpr {
namespace analysis {

// Returns true if the given expression is guaranteed to be positive.
bool mustBePositive(ExprPtr e) {
  if (e->isConstant()) {
    int e_val = immediateAs<int>(e);
    return e_val > 0;
  }
  return false;
}

OverlapKind boundOverlap(Bound a, Bound b) {
  // If they're equal they're equal.
  bool startEqual = exprEquals(a.start, b.start);
  bool endEqual = exprEquals(a.end, b.end);
  if (startEqual && endEqual) {
    return ContainedOrEqual;
  }

  // We have to figure out if the bounds fall under the following 2 cases:
  // 1. a is before b
  //      a.start ... a.end ... b.start ... b.end
  // 2. b is before a
  //      b.start ... b.end ... a.start ... a.end
  //
  // So, we compute "a.start - b.end" and "b.start - a.end". If even one of
  // those is positive, then it is guaranteed that the bounds do not overlap.
  //
  // If the diff is a constant, then we can directly check if the constant is
  // positive. If the diff is not a constant, then it will be made of
  // variables that correspond to the bounds of buffers involved. These buffer
  // bounds can never be negative. So, we check if the given expression is
  // guaranteed to be positive under the assumption that the variables involved
  // are never negative.

  ExprPtr lowDiff = IRSimplifier::simplify(alloc<Sub>(a.start, b.end));
  ExprPtr highDiff = IRSimplifier::simplify(alloc<Sub>(b.start, a.end));

  if (mustBePositive(lowDiff)) {
    return NoOverlap;
  }
  if (mustBePositive(highDiff)) {
    return NoOverlap;
  }

  ExprPtr diff_start = IRSimplifier::simplify(alloc<Sub>(b.start, a.start));
  ExprPtr diff_end = IRSimplifier::simplify(alloc<Sub>(b.end, a.end));

  // If one side fully encloses the other, they're adjacent.
  if (diff_start->isConstant() && diff_end->isConstant()) {
    int start = immediateAs<int>(diff_start);
    int end = immediateAs<int>(diff_end);
    // If diff_start and diff_end have different signs they are enclosing.
    if (start <= 0 && end >= 0) {
      return ContainedOrEqual;
    }

    if (start >= 0 && end <= 0) {
      return Contains;
    }
  }

  // We can't be sure there's no overlap so the conservative answer is
  // partial.
  return PartialOverlap;
}

bool indexBoundsEquals(const IndexBounds& A, const IndexBounds& B) {
  if (A.size() != B.size()) {
    return false;
  }

  for (size_t i = 0; i != A.size(); ++i) {
    if (!A[i].equals(B[i])) {
      return false;
    }
  }
  return true;
}

Bound flattenBounds(const IndexBounds& a) {
  if (a.empty()) {
    return Bound();
  }
  Bound ret = a[0];

  for (size_t i = 1; i < a.size(); ++i) {
    ret.start = alloc<Mul>(ret.start, a[i].start);
    ret.end = alloc<Mul>(ret.end, a[i].end);
  }

  ret.start = IRSimplifier::simplify(ret.start);
  ret.end = IRSimplifier::simplify(ret.end);
  return ret;
}

OverlapKind overlaps(const IndexBounds& a, const IndexBounds& b) {
  if (a.empty() && b.empty()) {
    return ContainedOrEqual;
  }

  // All accesses to a buf must have the same dimensionality.

  if (a.size() != b.size()) {
    return boundOverlap(flattenBounds(a), flattenBounds(b));
  }
  TORCH_INTERNAL_ASSERT(a.size() == b.size());

  OverlapKind overlap = boundOverlap(a[0], b[0]);
  for (size_t i = 1; i < a.size(); ++i) {
    OverlapKind bOverlap = boundOverlap(a[i], b[i]);
    if (bOverlap == NoOverlap) {
      return NoOverlap;
    }

    if (overlap == ContainedOrEqual && bOverlap == Contains) {
      overlap = Contains;
    }

    if (overlap == Contains && bOverlap == ContainedOrEqual) {
      continue;
    }

    if (bOverlap != overlap) {
      overlap = PartialOverlap;
      break;
    }
  }

  return overlap;
}

std::vector<Bound> subtractBound(Bound a, Bound b) {
  OverlapKind overlap = boundOverlap(a, b);
  if (overlap == NoOverlap) {
    return {a};
  }
  if (overlap == ContainedOrEqual) {
    return {};
  }

  // The bounds must overlap.
  std::vector<Bound> res;

  if (a.start->isConstant() != b.start->isConstant() ||
      a.end->isConstant() != b.end->isConstant()) {
    return {a};
  }

  ExprPtr lowDiff = IRSimplifier::simplify(alloc<Sub>(b.start, a.start));
  ExprPtr highDiff = IRSimplifier::simplify(alloc<Sub>(b.end, a.end));

  // If the diff has only a single var, we can try to guess sign.
  if (!lowDiff->isConstant()) {
    auto vars = VarFinder::find(lowDiff);
    if (vars.size() == 1) {
      lowDiff = IRSimplifier::simplify(alloc<Sub>(
          SubstituteInClone(b.start, {{*vars.begin(), immLike(b.start, 1)}}),
          SubstituteInClone(a.start, {{*vars.begin(), immLike(a.start, 1)}})));
    }
  }

  if (!highDiff->isConstant()) {
    auto vars = VarFinder::find(highDiff);
    if (vars.size() == 1) {
      highDiff = IRSimplifier::simplify(alloc<Sub>(
          SubstituteInClone(b.end, {{*vars.begin(), immLike(b.end, 1)}}),
          SubstituteInClone(a.end, {{*vars.begin(), immLike(a.end, 1)}})));
    }
  }

  bool hasHead = lowDiff->isConstant() && immediateAs<int>(lowDiff) > 0;
  bool hasTail = highDiff->isConstant() && immediateAs<int>(highDiff) < 0;

  bool constantExtents = lowDiff->isConstant() && highDiff->isConstant();

  if (!constantExtents) {
    // If we can't infer the bound lengths, there's no way to create a safe
    // subset. Just bail out.
    return {a};
  }

  if (hasHead) {
    res.emplace_back(
        a.start,
        IRSimplifier::simplify(alloc<Sub>(b.start, immLike(b.start, 1))));
  }

  if (hasTail) {
    ExprPtr tailStart =
        IRSimplifier::simplify(alloc<Add>(b.end, immLike(b.end, 1)));
    res.emplace_back(tailStart, a.end);
  }

  return res;
}

std::vector<IndexBounds> subtractIndicesBounds(
    const IndexBounds& A,
    const IndexBounds& B,
    OverlapKind overlap) {
  if (overlap == NoOverlap) {
    return {A};
  }

  if (overlap == ContainedOrEqual) {
    return {};
  }
  // All accesses to a buf must have the same dimensionality.
  TORCH_INTERNAL_ASSERT(A.size() == B.size(), buildErrorMessage());

  // Each dimension can be sliced into multiple bound segments.
  std::vector<IndexBounds> boundSlices;
  std::vector<Bound> remainingOuterBounds;

  for (size_t i = 0; i < A.size(); ++i) {
    auto slices = subtractBound(A[i], B[i]);

    Bound remaining = A[i];

    for (auto slice : slices) {
      IndexBounds newRegion;
      newRegion.reserve(A.size());
      TORCH_INTERNAL_ASSERT(
          remainingOuterBounds.size() == i, buildErrorMessage());

      for (size_t j = 0; j < i; ++j) {
        newRegion.push_back(remainingOuterBounds[j]);
      }
      newRegion.push_back(slice);
      for (size_t j = i + 1; j < A.size(); ++j) {
        newRegion.push_back(A[j]);
      }

      boundSlices.push_back(newRegion);

      if (slice.equals(A[i])) {
        remaining = A[i];
      } else {
        auto remainingSlices = subtractBound(remaining, slice);
        // In some cases, we might end up with empty remainingSlices due to the
        // optimization done in subtraction while handling diff expressions
        // that have a single variable in `subtractBound()`.
        if (!remainingSlices.empty()) {
          TORCH_INTERNAL_ASSERT(
              remainingSlices.size() == 1, buildErrorMessage());
          remaining = remainingSlices[0];
        }
      }
    }

    remainingOuterBounds.push_back(remaining);
  }

  return boundSlices;
}

std::vector<IndexBounds> TORCH_API
subtractIndicesBounds(const IndexBounds& A, const IndexBounds& B) {
  return subtractIndicesBounds(A, B, overlaps(A, B));
}

} // namespace analysis
} // namespace tensorexpr
} // namespace jit
} // namespace torch

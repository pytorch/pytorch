#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>

#include <iostream>

namespace torch::jit::tensorexpr::analysis {

// Returns true if the given expression is guaranteed to be positive.
static bool mustBePositive(const ExprPtr& e) {
  if (e->isConstant()) {
    int e_val = immediateAs<int>(e);
    return e_val > 0;
  }
  return false;
}

// Returns true if the given expression is guaranteed to be negative.
static bool mustBeNegative(const ExprPtr& e) {
  if (e->isConstant()) {
    int e_val = immediateAs<int>(e);
    return e_val < 0;
  }
  return false;
}

// Returns true if the given expression is guaranteed to be zero.
static bool mustBeZero(const ExprPtr& e) {
  if (e->isConstant()) {
    int e_val = immediateAs<int>(e);
    return e_val == 0;
  }
  return false;
}

void Bound::print() const {
  std::cout << "(" << *start << ", " << *end << ")";
}

bool Bound::equals(const Bound& other) const {
  return exprEquals(start, other.start) && exprEquals(end, other.end);
}

bool Bound::operator==(const Bound& other) const {
  if (equals(other)) {
    auto ret_expr = IRSimplifier::simplify(alloc<Sub>(start, end));
    return mustBeZero(ret_expr);
  }

  return false;
}

bool Bound::operator!=(const Bound& other) const {
  return (*this < other) || (*this > other);
}

bool Bound::operator>=(const Bound& other) const {
  if (*this == other) {
    return true;
  }
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(start, other.end));
  return mustBePositive(ret_expr) || mustBeZero(ret_expr);
}

bool Bound::operator>(const Bound& other) const {
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(start, other.end));
  return mustBePositive(ret_expr);
}

bool Bound::operator<=(const Bound& other) const {
  if (*this == other) {
    return true;
  }
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(end, other.start));
  return mustBeNegative(ret_expr) || mustBeZero(ret_expr);
}

bool Bound::operator<(const Bound& other) const {
  auto ret_expr = IRSimplifier::simplify(alloc<Sub>(end, other.start));
  return mustBeNegative(ret_expr);
}

OverlapKind boundOverlap(const Bound& a, const Bound& b) {
  // If they're equal they're equal.
  bool startEqual = exprEquals(a.start, b.start);
  bool endEqual = exprEquals(a.end, b.end);
  if (startEqual && endEqual) {
    return OverlapKind::ContainedOrEqual;
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
    return OverlapKind::NoOverlap;
  }
  if (mustBePositive(highDiff)) {
    return OverlapKind::NoOverlap;
  }

  ExprPtr diff_start = IRSimplifier::simplify(alloc<Sub>(b.start, a.start));
  ExprPtr diff_end = IRSimplifier::simplify(alloc<Sub>(b.end, a.end));

  // If one side fully encloses the other, they're adjacent.
  if (diff_start->isConstant() && diff_end->isConstant()) {
    int start = immediateAs<int>(diff_start);
    int end = immediateAs<int>(diff_end);
    // If diff_start and diff_end have different signs they are enclosing.
    if (start <= 0 && end >= 0) {
      return OverlapKind::ContainedOrEqual;
    }

    if (start >= 0 && end <= 0) {
      return OverlapKind::Contains;
    }
  }

  // We can't be sure there's no overlap so the conservative answer is
  // partial.
  return OverlapKind::PartialOverlap;
}

CmpEvalResult TORCH_API compareBound(
    const Bound& a,
    const Bound& b,
    const CompareSelectOperation& cmp_op) {
  switch (cmp_op) {
    case CompareSelectOperation::kGT:
      return (a > b)
          ? CmpEvalResult::True
          : (a <= b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kGE:
      return (a >= b)
          ? CmpEvalResult::True
          : (a < b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kLT:
      return (a < b)
          ? CmpEvalResult::True
          : (a >= b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kLE:
      return (a <= b)
          ? CmpEvalResult::True
          : (a > b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    case CompareSelectOperation::kNE:
      return (a != b)
          ? CmpEvalResult::True
          : (a == b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
    default:
      TORCH_INTERNAL_ASSERT(cmp_op == CompareSelectOperation::kEQ)
      return (a == b)
          ? CmpEvalResult::True
          : (a != b ? CmpEvalResult::False : CmpEvalResult::NotDetermined);
  }
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
    return OverlapKind::ContainedOrEqual;
  }

  // All accesses to a buf must have the same dimensionality.

  if (a.size() != b.size()) {
    return boundOverlap(flattenBounds(a), flattenBounds(b));
  }
  TORCH_INTERNAL_ASSERT(a.size() == b.size());

  OverlapKind overlap = boundOverlap(a[0], b[0]);
  for (size_t i = 1; i < a.size(); ++i) {
    OverlapKind bOverlap = boundOverlap(a[i], b[i]);
    if (bOverlap == OverlapKind::NoOverlap) {
      return OverlapKind::NoOverlap;
    }

    if (overlap == OverlapKind::ContainedOrEqual &&
        bOverlap == OverlapKind::Contains) {
      overlap = OverlapKind::Contains;
    }

    if (overlap == OverlapKind::Contains &&
        bOverlap == OverlapKind::ContainedOrEqual) {
      continue;
    }

    if (bOverlap != overlap) {
      overlap = OverlapKind::PartialOverlap;
      break;
    }
  }

  return overlap;
}

std::vector<Bound> subtractBound(const Bound& a, const Bound& b) {
  OverlapKind overlap = boundOverlap(a, b);
  if (overlap == OverlapKind::NoOverlap) {
    return {a};
  }
  if (overlap == OverlapKind::ContainedOrEqual) {
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
  if (overlap == OverlapKind::NoOverlap) {
    return {A};
  }

  if (overlap == OverlapKind::ContainedOrEqual) {
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

    for (const auto& slice : slices) {
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

} // namespace torch::jit::tensorexpr::analysis

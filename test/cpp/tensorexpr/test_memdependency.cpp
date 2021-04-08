#include <gtest/gtest.h>
#include <test/cpp/tensorexpr/test_base.h>

#include <torch/csrc/jit/tensorexpr/bounds_overlap.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// Test helper function used to determine if two regions of a buffer have an
// overlap. No Overlap & partial overlap is obvious. Contains means A is
// larger and fully encloses B, while ContainedOrEqual is the reverse. Equal
// ranges are ContainedOrEqual.
TEST(MemDependency, BoundOverlap) {
  KernelScope kernel_scope;

  using namespace analysis;

  auto CB = [](int s, int e) { return Bound(new IntImm(s), new IntImm(e)); };

  // Sanity check 3 overlap cases.
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(0, 0), CB(0, 0)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 3), CB(2, 5)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 0), CB(1, 1)));

  // Partial overlap works in either order.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 10), CB(7, 14)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(7, 14), CB(0, 10)));

  // Total Overlap works when one bound encloses the other, and returns which.
  ASSERT_EQ(Contains, boundOverlap(CB(2, 15), CB(7, 9)));
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(2, 15), CB(0, 16)));

  // Total overlap works when the bounds are an identical range, returns
  // ContainedOrEqual.
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(2, 15), CB(2, 15)));

  // Total overlap when only one end of the bound matches.
  ASSERT_EQ(Contains, boundOverlap(CB(2, 15), CB(2, 10)));
  ASSERT_EQ(Contains, boundOverlap(CB(2, 15), CB(3, 15)));
  ASSERT_EQ(Contains, boundOverlap(CB(0, 10), CB(0, 9)));
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(2, 10), CB(2, 15)));
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(3, 15), CB(2, 15)));

  // No overlap when a < b.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 2), CB(5, 10)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(2, 2), CB(3, 3)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(100, 120), CB(130, 130)));

  // No overlap when a > b.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(5, 10), CB(0, 2)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(3, 3), CB(2, 2)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(130, 130), CB(100, 120)));

  // No overlap when adjacent.
  ASSERT_EQ(NoOverlap, boundOverlap(CB(0, 100), CB(101, 120)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(2, 3), CB(0, 1)));

  // Partial overlap when middle bounds match.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 100), CB(100, 120)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(0, 2), CB(2, 4)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(100, 120), CB(0, 100)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(2, 3), CB(1, 2)));

  // Total overlap when one bound is single length over one end of the other.
  ASSERT_EQ(Contains, boundOverlap(CB(2, 15), CB(15, 15)));
  ASSERT_EQ(Contains, boundOverlap(CB(2, 15), CB(2, 2)));
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(2, 2), CB(2, 15)));
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(15, 15), CB(2, 15)));
}

TEST(MemDependency, BoundOverlapSymbolic) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  VarHandle w("w", kInt);

  using namespace analysis;

  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };

  // Sanity check cases where the start and end is symbolic but the diff is
  // constant.
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(x, x), CB(x, x)));
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, x + 3), CB(x + 2, x + 5)));
  ASSERT_EQ(NoOverlap, boundOverlap(CB(x, x), CB(x + 1, x + 1)));

  // We can't infer the sign of y, so cannot tell whether adding y is larger or
  // smaller than y/2.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, x + y), CB(x, x + y / 2)));

  // No information about this bound, have to take the most conservative option:
  // there may be an overlap.
  ASSERT_EQ(PartialOverlap, boundOverlap(CB(x, y), CB(z, w)));

  // Math on opaque terms works.
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(x + w, y - z), CB(x + w, y - z)));
  // Even requiring simplification.
  ASSERT_EQ(ContainedOrEqual, boundOverlap(CB(x - w - w, y), CB(x - w * 2, y)));
}

// Tests the helper function for overlap of multi dimensional indices bounds.
// This uses boundOverlap on each dimension and return the "lowest" kind of
// overlap.
TEST(MemDependency, BoundOverlapMultiDim) {
  KernelScope kernel_scope;

  using namespace analysis;

  auto CB = [](int s, int e) { return Bound(new IntImm(s), new IntImm(e)); };

  // Sanity check one dimensional cases.
  ASSERT_EQ(ContainedOrEqual, overlaps({CB(0, 0)}, {CB(0, 0)}));
  ASSERT_EQ(NoOverlap, overlaps({CB(0, 2)}, {CB(5, 10)}));
  ASSERT_EQ(PartialOverlap, overlaps({CB(0, 100)}, {CB(100, 120)}));

  // Total overlap in 3 dims.
  ASSERT_EQ(
      ContainedOrEqual,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 5), CB(0, 4)}));
  ASSERT_EQ(
      ContainedOrEqual,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 5), CB(0, 10)}));

  // Total overlap in 2 dims, no overlap in another.
  ASSERT_EQ(
      NoOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 5), CB(5, 10)}));

  // Total overlap in 2 dims, partial overlap in another.
  ASSERT_EQ(
      PartialOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(0, 2), CB(0, 5), CB(5, 10)}));
  // This case is most important, so verify the overlap in any dim. (dim 2)
  ASSERT_EQ(
      PartialOverlap,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(0, 2), CB(2, 6), CB(0, 5)}));
  // Dim 1.
  ASSERT_EQ(
      PartialOverlap,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(1, 3), CB(0, 5), CB(0, 5)}));
  // Total overlap in 1 dim, partial in 2.
  ASSERT_EQ(
      PartialOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(2, 6), CB(0, 5), CB(5, 10)}));
  // Total overlap, partial overlap, no overlap.
  ASSERT_EQ(
      NoOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(2, 6), CB(11, 15), CB(0, 5)}));

  // Total overlap (B) in 2 dims, total overlap (A) in another.
  ASSERT_EQ(
      Contains,
      overlaps({CB(0, 2), CB(0, 5), CB(0, 4)}, {CB(0, 2), CB(0, 3), CB(0, 4)}));

  // Total overlap (A) in 2 dims, total overlap (B) in another.
  ASSERT_EQ(
      Contains,
      overlaps(
          {CB(0, 12), CB(0, 15), CB(0, 4)}, {CB(0, 2), CB(0, 3), CB(0, 14)}));

  // Total (B), No Overlap, Total (A).
  ASSERT_EQ(
      NoOverlap,
      overlaps(
          {CB(0, 2), CB(0, 5), CB(0, 5)}, {CB(0, 6), CB(11, 15), CB(1, 2)}));
}

// Test the helper we use to subtract bounds: returns the regions(s) of A which
// remain after removing the region of B.
TEST(MemDependency, BoundSubtract) {
  KernelScope kernel_scope;

  using namespace analysis;

  auto CB = [](int s, int e) { return Bound(new IntImm(s), new IntImm(e)); };
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  // One element subtract.
  ASSERT_EQ(subtractBound(CB(0, 0), CB(0, 0)).size(), 0);
  ASSERT_EQ(subtractBound(CB(5, 5), CB(5, 5)).size(), 0);

  // No Overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5), CB(2, 2)), {CB(5, 5)}));
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5), CB(0, 4)), {CB(5, 5)}));

  // one side overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(4, 7)), {CB(1, 3)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 5), CB(5, 7)), {CB(0, 4)}));
  ASSERT_TRUE(EQ(subtractBound(CB(4, 5), CB(1, 4)), {CB(5, 5)}));
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(0, 4)), {CB(5, 5)}));

  // both sides overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(0, 7)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(5, 5), CB(5, 7)), {}));

  // internal overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(1, 5), CB(2, 3)), {CB(1, 1), CB(4, 5)}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, 5), CB(2, 4)), {CB(0, 1), CB(5, 5)}));
}

TEST(MemDependency, BoundSubtractSymbolic) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  VarHandle w("w", kInt);

  using namespace analysis;

  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  // One element subtract.
  ASSERT_TRUE(EQ(subtractBound(CB(x, x), CB(x, x)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x + 1, x + 1), CB(x + 1, x + 1)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x * 2, x * 2), CB(x * 2, x * 2)), {}));

  // Subtract constant range low.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10), CB(x, x + 4)), {CB(x + 5, x + 10)}));
  // Subtract constant range high.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10), CB(x + 6, x + 12)), {CB(x, x + 5)}));
  // Subtract constant range total overlap.
  ASSERT_TRUE(EQ(subtractBound(CB(x, x + 10), CB(x, x + 10)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(x + 2, x + 10), CB(x, x + 12)), {}));
  // Subtract constant range internal.
  ASSERT_TRUE(
      EQ(subtractBound(CB(x, x + 10), CB(x + 3, x + 7)),
         {CB(x, x + 2), CB(x + 8, x + 10)}));

  // Size is inferable but not constant, only works with a single var.
  ASSERT_TRUE(EQ(subtractBound(CB(0, x), CB(0, x * 2)), {}));
  ASSERT_TRUE(EQ(subtractBound(CB(0, x * 2), CB(0, x - 1)), {CB(x, x * 2)}));

  // Size is not inferable.
  ASSERT_TRUE(EQ(subtractBound(CB(x, y), CB(z, w)), {CB(x, y)}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, y), CB(x, z)), {CB(x, y)}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, y), CB(0, x)), {CB(x, y)}));
  ASSERT_TRUE(EQ(subtractBound(CB(x, x), CB(0, 0)), {CB(x, x)}));
}

// Tests the helper function that does subtraction, but for multi dimensional
// indices bounds.
TEST(MemDependency, BoundSubtractMultiDim) {
  KernelScope kernel_scope;

  using namespace analysis;

  auto CB = [](int s, int e) { return Bound(new IntImm(s), new IntImm(e)); };
  auto EQ = [](std::vector<IndexBounds> x, std::vector<IndexBounds> y) {
    if (x.size() != y.size()) {
      return false;
    }
    for (auto i = 0; i < x.size(); ++i) {
      if (!indexBoundsEquals(x[i], y[i])) {
        return false;
      }
    }
    return true;
  };

  // sanity check one dimension.
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(0, 9)}, {CB(0, 9)}), {}));
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(3, 9)}, {CB(0, 12)}), {}));
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 12)}, {CB(0, 9)}), {{CB(10, 12)}}));
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 12)}, {CB(3, 12)}), {{CB(0, 2)}}));
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(0, 9)}, {CB(1, 8)}), {{CB(0, 0)}, {CB(9, 9)}}));

  // Multi dim total overlap.
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(0, 9), CB(0, 2)}, {CB(0, 9), CB(0, 2)}), {}));
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(0, 9), CB(0, 2)}, {CB(0, 10), CB(0, 20)}), {}));

  // Mutli dim one way partial in dim 1.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 9), CB(0, 2)}, {CB(0, 3), CB(0, 2)}),
         {{CB(4, 9), CB(0, 2)}}));

  // Mutli dim one way partial in dim 2.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 9), CB(0, 20)}, {CB(0, 9), CB(0, 10)}),
         {{CB(0, 9), CB(11, 20)}}));

  // Partial overlap in 2 dims.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, 5), CB(0, 5)}, {CB(2, 8), CB(2, 8)}),
         {{CB(0, 1), CB(0, 5)}, {CB(2, 5), CB(0, 1)}}));

  // Partial overlap in 3 dims.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds(
             {CB(0, 5), CB(0, 5), CB(0, 5)}, {CB(2, 8), CB(2, 8), CB(2, 8)}),
         {{CB(0, 1), CB(0, 5), CB(0, 5)},
          {CB(2, 5), CB(0, 1), CB(0, 5)},
          {CB(2, 5), CB(2, 5), CB(0, 1)}}));
}

// Tests the multi dimensional subtraction code for bounds that cannot be fully
// materialized.
TEST(MemDependency, BoundSubtractMultiDimSymbolic) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  using namespace analysis;

  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };

  auto EQ = [](std::vector<IndexBounds> x, std::vector<IndexBounds> y) {
    if (x.size() != y.size()) {
      return false;
    }
    for (auto i = 0; i < x.size(); ++i) {
      if (!indexBoundsEquals(x[i], y[i])) {
        return false;
      }
    }
    return true;
  };

  // Cannot determine overlaps.
  ASSERT_TRUE(EQ(subtractIndicesBounds({CB(x, x)}, {CB(0, 0)}), {{CB(x, x)}}));

  // Various total Overlaps.
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(x, x), CB(x, x)}, {CB(x, x), CB(x, x)}), {}));
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(x, y), CB(x, y)}, {CB(x, y), CB(x, y)}), {}));
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(x, x), CB(y, y)}, {CB(x, x), CB(y, y)}), {}));
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x), CB(0, y)}), {}));

  // one-way overlap in first dim.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x - 5), CB(0, y)}),
         {{CB(x - 4, x), CB(0, y)}}));
  // second dim.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x), CB(5, y)}),
         {{CB(0, x), CB(0, 4)}}));

  // Internal overlap in first dim.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(2, x - 5), CB(0, y)}),
         {{CB(0, 1), CB(0, y)}, {CB(x - 4, x), CB(0, y)}}));
  // second dim.
  ASSERT_TRUE(EQ(
      subtractIndicesBounds({CB(0, x), CB(0, y)}, {CB(0, x), CB(10, y - 10)}),
      {{CB(0, x), CB(0, 9)}, {CB(0, x), CB(y - 9, y)}}));

  // Overlap in both dimensions.
  ASSERT_TRUE(
      EQ(subtractIndicesBounds(
             {CB(0, x), CB(0, y)}, {CB(5, x - 5), CB(10, y - 10)}),
         {
             {CB(0, 4), CB(0, y)},
             {CB(x - 4, x), CB(0, y)},
             {CB(0, x), CB(0, 9)},
             {CB(0, x), CB(y - 9, y)},
         }));
}

// Simple check that the analyzer does anything at all...
TEST(MemDependency, MemDependencyCheckerSimple) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);

  analysis::MemDependencyChecker analyzer;

  /*
   * A[0] = 3;
   * B[0] = A[0] + 1;
   */

  Store* aStore = Store::make(a, {0}, 3, 1);
  Store* bStore = Store::make(b, {0}, Add::make(Load::make(a, {0}, 1), 1), 1);

  Stmt* stmt = Block::make({aStore, bStore});

  stmt->accept(&analyzer);

  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aStore));
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, bStore));
  // sanity check, but anything that depends directly must depend indirectly.
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, aStore));
}

// Check that there is a difference between direct and indirect dependence.
TEST(MemDependency, MemDependencyCheckerMultiStmt) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  BufHandle c("C", {1}, kInt);

  analysis::MemDependencyChecker analyzer;

  /*
   * A[0] = 3;
   * B[0] = A[0];
   * C[0] = B[0] + 1;
   */

  Store* aStore = Store::make(a, {0}, 3, 1);
  Store* bStore = Store::make(b, {0}, Load::make(a, {0}, 1), 1);
  Store* cStore = Store::make(c, {0}, Add::make(Load::make(b, {0}, 1), 1), 1);

  Stmt* stmt = Block::make({aStore, bStore, cStore});

  stmt->accept(&analyzer);

  // C depends on A indirectly.
  ASSERT_FALSE(analyzer.dependsDirectly(cStore, aStore));
  ASSERT_TRUE(analyzer.dependsIndirectly(cStore, aStore));

  // C depends on B directly, which depends on A directly.
  ASSERT_TRUE(analyzer.dependsDirectly(cStore, bStore));
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aStore));

  // Dependency goes top to bottom only.
  ASSERT_FALSE(analyzer.dependsIndirectly(bStore, cStore));
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, bStore));
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, cStore));
}

// Verify that we do filter writes that are totally overlapped by later writes.
TEST(MemDependency, MemDependencyCheckerOverlap) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);

  analysis::MemDependencyChecker analyzer;

  /*
   * A[0] = 3;
   * A[0] = 6;
   * B[0] = A[0] + 1;
   */

  Store* aStore = Store::make(a, {0}, 3, 1);
  Store* a2Store = Store::make(a, {0}, 6, 1);
  Store* bStore = Store::make(b, {0}, Add::make(Load::make(a, {0}, 1), 1), 1);

  Stmt* stmt = Block::make({aStore, a2Store, bStore});

  stmt->accept(&analyzer);

  // B store depends on second A store but not first since it is completely
  // overlapped.
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, a2Store));
  ASSERT_FALSE(analyzer.dependsIndirectly(bStore, aStore));

  // No dependency between either A store.
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, a2Store));
  ASSERT_FALSE(analyzer.dependsIndirectly(a2Store, aStore));
}

// Verify that bounds match loop iterations, and that dependencies progress
// across loop scopes.
TEST(MemDependency, MemDependencyCheckerLoop) {
  KernelScope kernel_scope;
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  VarHandle x("x", kInt);

  using namespace analysis;

  MemDependencyChecker analyzer;

  /*
   * for (int x = 0; x < 10; ++x) {
   *   A[x] = x;
   * }
   * B[0] = A[0] + 1;
   */

  Store* aStore = Store::make(a, {x}, x, 1);
  Stmt* loop = For::make(x, 0, 10, aStore);
  Store* bStore = Store::make(b, {0}, Add::make(Load::make(a, {4}, 1), 1), 1);

  Stmt* stmt = Block::make({loop, bStore});

  stmt->accept(&analyzer);

  // Same A->B dependency.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aStore));

  // B depends on the loop.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, loop));
  // A is in the loop but does not depend on any loop iteration.
  ASSERT_FALSE(analyzer.dependsIndirectly(aStore, loop));

  auto aStoreAccess = analyzer.accessFor(aStore);
  ASSERT_NE(aStoreAccess, nullptr);

  // It should have bounds covering the range of x: 0 <= x < 10.
  ASSERT_TRUE(indexBoundsEquals(
      aStoreAccess->bounds(), {Bound(new IntImm(0), new IntImm(9))}));
}

// Reductions should promote dependencies as well.
TEST(MemDependency, MemDependencyCheckerLoopReduce) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  using namespace analysis;

  MemDependencyChecker analyzer;

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; ++x) {
   *   A[0] = A[x] + 1;
   * }
   * B[0] = A[0];
   */

  Store* aInit = Store::make(a, {0}, 0, 1);
  ExprHandle reduce =
      ExprHandle(Sum()(a.node(), ExprHandle(1), {x.node()}, {x.node()}));
  Store* aReduce = Store::make(a, {0}, reduce, 1);
  Stmt* loop = For::make(x, 0, 10, aReduce);
  Store* bStore = Store::make(b, {0}, Load::make(a, {0}, 1), 1);

  Stmt* stmt = Block::make({aInit, loop, bStore});

  stmt->accept(&analyzer);

  // B -> A.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aReduce));

  // B depends indirectly on the intializer of A, since the reduction depends
  // on it.
  ASSERT_FALSE(analyzer.dependsDirectly(bStore, aInit));
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, aInit));

  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, aInit));

  // B depends on the loop.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, loop));
  // A is in the loop and depends on other iterations.
  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, loop));

  // The loop contents depend on the initializer too.
  ASSERT_TRUE(analyzer.dependsDirectly(loop, aInit));

  // Find loads within the reduction:
  auto reduceLoads = NodeFinder<Load>::find(reduce.node());
  // Pull out the access for the load inside the loop.
  for (auto* load : reduceLoads) {
    auto loopLoad = analyzer.accessFor(load);
    // It should have 10 element long bounds.
    ASSERT_TRUE(indexBoundsEquals(
        loopLoad->bounds(), {Bound(new IntImm(0), new IntImm(9))}));
  }
}

// Lowering a reduction doesn't affect dependency analysis.
TEST(MemDependency, MemDependencyCheckerLoopReduceExpanded) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  using namespace analysis;

  MemDependencyChecker analyzer;

  /*
   * A[0] = 0;
   * for (int x = 0; x < 10; ++x) {
   *   A[0] = A[x] + 1;
   * }
   * B[0] = A[0];
   */

  Store* aInit = Store::make(a, {0}, 0, 1);
  ExprHandle aLoad = Load::make(a, {x}, 1);
  Store* aReduce = Store::make(a, {0}, Add::make(aLoad, 1));
  Stmt* loop = For::make(x, 0, 10, aReduce);
  Store* bStore = Store::make(b, {0}, Load::make(a, {0}, 1), 1);

  Stmt* stmt = Block::make({aInit, loop, bStore});

  stmt->accept(&analyzer);

  // B -> A.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, aReduce));

  // B depends indirectly on the intializer of A, since the reduction depends
  // on it.
  ASSERT_FALSE(analyzer.dependsDirectly(bStore, aInit));
  ASSERT_TRUE(analyzer.dependsIndirectly(bStore, aInit));

  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, aInit));

  // B depends on the loop.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, loop));
  // A is in the loop and depends on other iterations.
  ASSERT_TRUE(analyzer.dependsDirectly(aReduce, loop));

  // The loop contents depend on the initializer too.
  ASSERT_TRUE(analyzer.dependsDirectly(loop, aInit));

  // Pull out the access for the store inside the loop.
  auto loopLoad = analyzer.accessFor(aLoad.node());
  // It should have 10 element long bounds.
  ASSERT_TRUE(indexBoundsEquals(
      loopLoad->bounds(), {Bound(new IntImm(0), new IntImm(9))}));
}

// Can determine dependencies of outputs, through to inputs.
TEST(MemDependency, MemDependencyCheckerInputsOutputs) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  // initialize analyzer with inputs and outputs.
  analysis::MemDependencyChecker analyzer({a}, {b});

  // Here's a Relu.
  /*
   * for (int x = 0; x < 10; ++x) {
   *   B[x] = Max(A[x], 0);
   * }
   */

  ExprHandle aLoad = Load::make(a, {x}, 1);
  Store* bStore = Store::make(b, {x}, Max::make(aLoad, 0, true), 1);
  Stmt* loop = For::make(x, 0, 10, bStore);

  Stmt* stmt = Block::make({loop});

  stmt->accept(&analyzer);

  // Output depends indirectly on input.
  ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));
  // aLoad depends directly on the input A.
  ASSERT_TRUE(analyzer.dependsDirectly(aLoad.node(), a.node()));
  // bStore therefore depends directly on the input A.
  ASSERT_TRUE(analyzer.dependsDirectly(bStore, a.node()));
  // The output depends directly on the store.
  ASSERT_TRUE(analyzer.dependsDirectly(b.node(), bStore));

  // Check AccessInfo based overloads.
  auto input = analyzer.input(a.node());
  auto output = analyzer.output(b.node());

  // Output depends indirectly on input.
  ASSERT_TRUE(analyzer.dependsIndirectly(output, input));
  // Not directly.
  ASSERT_FALSE(analyzer.dependsDirectly(output, input));
  // Not in reverse order.
  ASSERT_FALSE(analyzer.dependsIndirectly(input, output));

  // output -> bStore -> bLoad -> input.
  auto storeAccess = analyzer.accessFor(bStore);
  auto loadAccess = analyzer.accessFor(aLoad.node());

  ASSERT_TRUE(analyzer.dependsDirectly(output, storeAccess));
  ASSERT_TRUE(analyzer.dependsDirectly(loadAccess, input));
}

// Can tell if an output does not depend on an input.
TEST(MemDependency, MemDependencyCheckerOutputDoesntDepend) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  // initialize analyzer with inputs and outputs.
  analysis::MemDependencyChecker analyzer({a}, {b});

  // Here's a dumb Relu.
  /*
   * for (int x = 0; x < 10; ++x) {
   *   B[x] = Max(x, 0);
   * }
   */

  Store* bStore = Store::make(b, {x}, Max::make(x, 0, true), 1);
  Stmt* loop = For::make(x, 0, 10, bStore);

  Stmt* stmt = Block::make({loop});

  stmt->accept(&analyzer);

  // Output does not depend indirectly on input.
  ASSERT_FALSE(analyzer.dependsIndirectly(b.node(), a.node()));

  // The output still depends directly on the store.
  ASSERT_TRUE(analyzer.dependsDirectly(b.node(), bStore));

  // Check AccessInfo based overloads.
  auto input = analyzer.input(a.node());
  auto output = analyzer.output(b.node());

  // Output does not depend indirectly on input.
  ASSERT_FALSE(analyzer.dependsIndirectly(output, input));
}

// Verify different loop extents produce accesses with different bounds, and
// that later accesses find dependencies that overlap their entire bound range.
TEST(MemDependency, MemDependencyCheckerLoopBounds) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  BufHandle c("C", {10}, kInt);
  VarHandle x("x", kInt);
  using namespace analysis;

  MemDependencyChecker analyzer({a}, {c});

  // This enables using the execution order of the loops to determine if some
  // loops are self dependent or not.
  analyzer.allowLoopExecutionOrderAnalysis();

  /*
   * for (int x = 1; x < 10; ++x) {
   *   B[x] = A[x];
   * }
   * for (int x = 1; x < 9; ++x) {
   *   B[x] = B[x] * 2;
   * }
   * for (int x = 3; x < 4; ++x) {
   *   C[x] = A[x];
   * }
   * for (int x = 0; x < 10; ++x) {
   *   C[x] = B[x];
   * }
   */

  std::vector<Stmt*> stmts(
      {For::make(x, 1, 10, Store::make(b, {x}, Load::make(a, {x}, 1), 1)),
       For::make(
           x,
           1,
           9,
           Store::make(b, {x}, Mul::make(Load::make(b, {x}, 1), 2), 1)),
       For::make(x, 3, 4, Store::make(c, {x}, Load::make(a, {x}, 1), 1)),
       For::make(x, 0, 10, Store::make(c, {x}, Load::make(b, {x}, 1), 1))});

  Stmt* stmt = Block::make(stmts);

  stmt->accept(&analyzer);

  auto input = analyzer.input(a.node());
  auto output = analyzer.output(c.node());

  // sanity check Output -> Input.
  ASSERT_TRUE(analyzer.dependsIndirectly(output, input));

  // Check the For loop dependencies:

  // Last write to C depends on both writes to B since they contain the last
  // write to at least one element.
  ASSERT_TRUE(analyzer.dependsIndirectly(stmts[3], stmts[1]));
  ASSERT_TRUE(analyzer.dependsIndirectly(stmts[3], stmts[0]));

  // The last write to C does not depend on the other write to C.
  ASSERT_FALSE(analyzer.dependsIndirectly(stmts[3], stmts[2]));

  auto CB = [](int s, int e) { return Bound(new IntImm(s), new IntImm(e)); };
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  /*  0. Input: A[(0, 9)] - dependents: 1 5
   *  1. Load: A[(1, 9)] - depends on: 0  - dependents: 2
   *  2. Store: B[(1, 9)] - depends on: 1  - dependents: 3 7
   *  3. Load: B[(1, 8)] - depends on: 2  - dependents: 4
   *  4. Store: B[(1, 8)] - depends on: 3  - dependents: 7
   *  5. Load: A[(3, 3)] - depends on: 0  - dependents: 6
   *  6. Store: C[(3, 3)] - depends on: 5
   *  7. Load: B[(0, 9)] - depends on: 2 4  - dependents: 8
   *  8. Store: C[(0, 9)] - depends on: 7  - dependents: 9
   *  9. Output: C[(0, 9)] - depends on: 8
   */

  // Now let's look at the bounds of each access.
  // There are 9 accesses in this Stmt, so this is exhaustive, we wont do this
  // much.
  auto history = analyzer.getHistory();
  ASSERT_EQ(history.size(), 10);
  const Var* aVar = a.node()->base_handle();
  const Var* bVar = b.node()->base_handle();
  const Var* cVar = c.node()->base_handle();

  // The first access is the input A.
  ASSERT_EQ(history[0]->type(), AccessType::Input);
  ASSERT_EQ(history[0]->var(), aVar);
  // It has the bounds of the producing Input.
  ASSERT_TRUE(EQ(history[0]->bounds(), {CB(0, 9)}));
  // sanity check the input we retrieved earlier matches.
  ASSERT_EQ(history[0], input);

  // The second access is the load of A in the first loop.
  ASSERT_EQ(history[1]->type(), AccessType::Load);
  ASSERT_EQ(history[1]->var(), aVar);
  // It has the bounds of the loop, i.e. start == 1.
  ASSERT_TRUE(EQ(history[1]->bounds(), {CB(1, 9)}));
  // It reads from A, so it should have a dependency on the last write to this
  // range - with is the input.
  ASSERT_EQ(history[1]->dependencies().size(), 1);
  ASSERT_TRUE(history[1]->hasDependency(history[0]));

  // The third access is the store into B in the first loop.
  ASSERT_EQ(history[2]->type(), AccessType::Store);
  ASSERT_EQ(history[2]->var(), bVar);
  // It also has the bounds of the loop, i.e. start == 1.
  ASSERT_TRUE(EQ(history[2]->bounds(), {CB(1, 9)}));
  // The previous load is in its RHS, so it depends on it.
  ASSERT_EQ(history[2]->dependencies().size(), 1);
  ASSERT_TRUE(history[2]->hasDependency(history[1]));

  // The third access is the load from B in the second loop.
  ASSERT_EQ(history[3]->type(), AccessType::Load);
  ASSERT_EQ(history[3]->var(), bVar);
  // It has the bounds of the second loop, i.e. >= 1 < 9.
  ASSERT_TRUE(EQ(history[3]->bounds(), {CB(1, 8)}));
  // It reads from B in a smaller range, so should depend on the previous
  // store.
  ASSERT_EQ(history[3]->dependencies().size(), 1);
  ASSERT_TRUE(history[3]->hasDependency(history[2]));

  // The fourth: the store to B in the second loop.
  ASSERT_EQ(history[4]->type(), AccessType::Store);
  ASSERT_EQ(history[4]->var(), bVar);
  // It also has the bounds of the second loop.
  ASSERT_TRUE(EQ(history[4]->bounds(), {CB(1, 8)}));
  // The previous load is in its RHS, so it depends on it as before.
  ASSERT_EQ(history[4]->dependencies().size(), 1);
  ASSERT_TRUE(history[4]->hasDependency(history[3]));

  // The fifth access is the load is from the 3rd loop, and skips previous B
  // accesses.
  ASSERT_EQ(history[5]->type(), AccessType::Load);
  ASSERT_EQ(history[5]->var(), aVar);
  // It has the bounds of the third loop: >= 3 < 4.
  ASSERT_TRUE(EQ(history[5]->bounds(), {CB(3, 3)}));
  // It depends on the last thing to write to A, which is the A input.
  ASSERT_EQ(history[5]->dependencies().size(), 1);
  ASSERT_TRUE(history[5]->hasDependency(history[0]));

  // Sixth: the store into the output C.
  ASSERT_EQ(history[6]->type(), AccessType::Store);
  ASSERT_EQ(history[6]->var(), cVar);
  // It also has the bounds of the third loop.
  ASSERT_TRUE(EQ(history[6]->bounds(), {CB(3, 3)}));
  // The previous load is in its RHS, so it depends on it as always.
  ASSERT_EQ(history[6]->dependencies().size(), 1);
  ASSERT_TRUE(history[6]->hasDependency(history[5]));

  // The seventh access is the load of B in the fourth loop.
  ASSERT_EQ(history[7]->type(), AccessType::Load);
  ASSERT_EQ(history[7]->var(), bVar);
  // It has the bounds of the final loop, >= 0 < 10
  ASSERT_TRUE(EQ(history[7]->bounds(), {CB(0, 9)}));
  // The bounds of this read are larger than the bounds of the previous write,
  // so it depends on both previous Stores to B.
  ASSERT_EQ(history[7]->dependencies().size(), 2);
  ASSERT_TRUE(history[7]->hasDependency(history[2]));
  ASSERT_TRUE(history[7]->hasDependency(history[4]));

  // Eight: the final store into the output C.
  ASSERT_EQ(history[8]->type(), AccessType::Store);
  ASSERT_EQ(history[8]->var(), cVar);
  // It also has the bounds of the final loop.
  ASSERT_TRUE(EQ(history[8]->bounds(), {CB(0, 9)}));
  // The previous load is in its RHS, so it depends on it as always.
  ASSERT_EQ(history[8]->dependencies().size(), 1);
  ASSERT_TRUE(history[8]->hasDependency(history[7]));

  // The last access represents the output Buf.
  ASSERT_EQ(history[9]->type(), AccessType::Output);
  ASSERT_EQ(history[9]->var(), cVar);
  // It has the bounds of the output Buf.
  ASSERT_TRUE(EQ(history[9]->bounds(), {CB(0, 9)}));
  // sanity check the input we retrieved earlier matches.
  ASSERT_EQ(history[9], output);
  // It depends on the last write to C only.
  ASSERT_EQ(history[9]->dependencies().size(), 1);
  ASSERT_TRUE(history[9]->hasDependency(history[8]));
}

// Verify that we can still infer bounds when the loop var is offset.
TEST(MemDependency, MemDependencyCheckerLoopBoundsIndexShift) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  using namespace analysis;

  MemDependencyChecker analyzer({a}, {b});

  // This enables using the execution order of the loops to determine if some
  // loops are self dependent or not.
  analyzer.allowLoopExecutionOrderAnalysis();

  /*
   * for (int x = 1; x < 10; x++) {
   *   A[x] = A[x - 1];
   * }
   * for (int x = 0; x < 9; x++) {
   *   A[x] = A[x + 1];
   * }
   * for (int x = 0; x < 9; x++) {
   *   A[9 - x] = A[8 - x];
   * }
   * for (int x = 0; x < 10; x++) {
   *   A[x] = A[9 - x];
   * }
   * for (int x = 0; x < 10; x++) {
   *   B[x] = A[x];
   * }
   */

  Stmt* stmt = Block::make(
      {For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}, 1), 1)),
       For::make(x, 0, 9, Store::make(a, {x}, Load::make(a, {x + 1}, 1), 1)),
       For::make(
           x,
           0,
           9,
           Store::make(
               a,
               {ExprHandle(9) - x},
               Load::make(a, {ExprHandle(8) - x}, 1),
               1)),
       For::make(
           x,
           0,
           10,
           Store::make(a, {x}, Load::make(a, {ExprHandle(9) - x}, 1), 1)),
       For::make(x, 0, 10, Store::make(b, {x}, Load::make(a, {x}, 1), 1))});

  stmt->accept(&analyzer);

  // Sanity check output depends on Input.
  ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

  auto CB = [](int s, int e) { return Bound(new IntImm(s), new IntImm(e)); };
  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  /*  0. Input: A[(0, 9)] - dependents: 1
   *  1. Load: A[(0, 8)] - depends on: 0 2  - dependents: 2
   *  2. Store: A[(1, 9)] - depends on: 1  - dependents: 1 3
   *  3. Load: A[(1, 9)] - depends on: 2  - dependents: 4
   *  4. Store: A[(0, 8)] - depends on: 3  - dependents: 5 7
   *  5. Load: A[(0, 8)] - depends on: 4  - dependents: 6
   *  6. Store: A[(1, 9)] - depends on: 5  - dependents: 7
   *  7. Load: A[(0, 9)] - depends on: 4 6 8  - dependents: 8
   *  8. Store: A[(0, 9)] - depends on: 7  - dependents: 7 9
   *  9. Load: A[(0, 9)] - depends on: 8  - dependents: 10
   *  10. Store: B[(0, 9)] - depends on: 9  - dependents: 11
   *  11. Output: B[(0, 9)] - depends on: 10
   */

  // Now let's look at the bounds of each access.
  auto history = analyzer.getHistory();
  ASSERT_EQ(history.size(), 12);
  const Var* aVar = a.node()->base_handle();
  const Var* bVar = b.node()->base_handle();

  // The first access is the input A.
  ASSERT_EQ(history[0]->type(), AccessType::Input);
  ASSERT_EQ(history[0]->var(), aVar);
  // It has the bounds of the producing Input.
  ASSERT_TRUE(EQ(history[0]->bounds(), {CB(0, 9)}));

  // The second access is the load A[x-1].
  ASSERT_EQ(history[1]->type(), AccessType::Load);
  ASSERT_EQ(history[1]->var(), aVar);
  // It has the bounds of the loop modified by the offset of each index, in
  // this case -1.
  ASSERT_TRUE(EQ(history[1]->bounds(), {CB(0, 8)}));
  // It depends on the input, but also the store in the same loop, since
  // different interations of the loop depend on each other.
  ASSERT_EQ(history[1]->dependencies().size(), 2);
  ASSERT_TRUE(history[1]->hasDependency(history[0]));
  ASSERT_TRUE(history[1]->hasDependency(history[2]));

  // The third access is the Store to A[x] in the first loop.
  ASSERT_EQ(history[2]->type(), AccessType::Store);
  ASSERT_EQ(history[2]->var(), aVar);
  // It has no offset on x, so should have the same bounds as the loop.
  ASSERT_TRUE(EQ(history[2]->bounds(), {CB(1, 9)}));

  // The fourth access is the load A[x+1] in the second loop.
  ASSERT_EQ(history[3]->type(), AccessType::Load);
  ASSERT_EQ(history[3]->var(), aVar);
  // It has the bounds of the loop (0 <= x < 9) modified by the offset of each
  // index, in this case 1.
  ASSERT_TRUE(EQ(history[3]->bounds(), {CB(1, 9)}));
  // This load totally overlaps the previous write to A, so it depends only on
  // it and not the input.
  ASSERT_EQ(history[3]->dependencies().size(), 1);
  ASSERT_TRUE(history[3]->hasDependency(history[2]));

  // The fifth access is the store to A[x] in the second loop.
  ASSERT_EQ(history[4]->type(), AccessType::Store);
  ASSERT_EQ(history[4]->var(), aVar);
  // It has no offset on x, so should have the same bounds as the loop.
  ASSERT_TRUE(EQ(history[4]->bounds(), {CB(0, 8)}));

  // The sixth access is the load to A[8 - x] in the third loop.
  ASSERT_EQ(history[5]->type(), AccessType::Load);
  ASSERT_EQ(history[5]->var(), aVar);
  // It has the bounds of the loop (0 <= x < 9) modified by the offset of each
  // index, in this case 8 - x.
  // This access has a negative stride, which will be normalized.
  ASSERT_TRUE(EQ(history[5]->bounds(), {CB(0, 8)}));
  // This load totally overlaps the most recent write to A, so it depends only
  // on it and not the input or the first write to A.
  ASSERT_EQ(history[5]->dependencies().size(), 1);
  ASSERT_TRUE(history[5]->hasDependency(history[4]));

  // The seventh access is the store to A[9 - x] in the third loop.
  ASSERT_EQ(history[6]->type(), AccessType::Store);
  ASSERT_EQ(history[6]->var(), aVar);
  // This store has a negative stride on it's indices, but is notmalized
  // internally.
  ASSERT_TRUE(EQ(history[6]->bounds(), {CB(1, 9)}));

  // The eighth access is the load A[9-x] in the second loop.
  ASSERT_EQ(history[7]->type(), AccessType::Load);
  ASSERT_EQ(history[7]->var(), aVar);
  // It has the bounds of the loop (0 <= x < 9), modified by the offset 9 - x,
  // which esstentially traverses the loop backwards.
  ASSERT_TRUE(EQ(history[7]->bounds(), {CB(0, 9)}));
  // This Load has three write dependencies:
  ASSERT_EQ(history[7]->dependencies().size(), 3);
  //  * The previous store (#6) for elements 1-9
  ASSERT_TRUE(history[7]->hasDependency(history[6]));
  //  * An earlier store (#4) covering element 0
  ASSERT_TRUE(history[7]->hasDependency(history[4]));
  //  * A future store inside this loop, since this loop modifies the buffer
  //  in a non distinct way (due to the load and store having different access
  //  strides).
  ASSERT_TRUE(history[7]->hasDependency(history[8]));

  // The ninth access is the store to A[x] in the fourth loop.
  ASSERT_EQ(history[8]->type(), AccessType::Store);
  ASSERT_EQ(history[8]->var(), aVar);
  // This store has a negative stride on it's indices, but is notmalized
  // internally.
  ASSERT_TRUE(EQ(history[8]->bounds(), {CB(0, 9)}));

  // The tenth and 11th acceses are the copy from A[x] to B[x].
  ASSERT_EQ(history[9]->type(), AccessType::Load);
  ASSERT_EQ(history[9]->var(), aVar);
  ASSERT_EQ(history[10]->type(), AccessType::Store);
  ASSERT_EQ(history[10]->var(), bVar);

  // The last access represents the output Buf.
  ASSERT_EQ(history[11]->type(), AccessType::Output);
  ASSERT_EQ(history[11]->var(), bVar);
  // It has the bounds of the output Buf.
  ASSERT_TRUE(EQ(history[11]->bounds(), {CB(0, 9)}));
  // It depends on the last write to B only.
  ASSERT_EQ(history[11]->dependencies().size(), 1);
  ASSERT_TRUE(history[11]->hasDependency(history[10]));

  // ok that's enough of that.
}

// Check many different cases of loop self dependency - when a load within a
// loop is dependent on a Store later in the same loop but in different
// iteration. This is affected by whether or not we can trust the execution
// order of the loop.
TEST(MemDependency, MemDependencyCheckerLoopSelfDependency) {
  KernelScope kernel_scope;
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  using namespace analysis;

  // This check assumes that the Stmt has a single Store with a single Load on
  // the RHS.
  auto isSelfDependent =
      [](const std::vector<std::shared_ptr<AccessInfo>>& history) -> bool {
    return history.front()->hasDependency(history.back());
  };

  {
    /* for (int y = 0; y < 10; y++) {
     *   A[y] = (A[y]) + 1;
     * } */

    // Not self dependent since all loop iterations use a different y.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        y,
        0,
        10,
        Block::make(
            {Store::make(a, {y}, Add::make(Load::make(a, {y}, 1), 1), 1)}));

    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int y = 0; y < 10; y++) {
     *   A[y + 1] = (A[y + 1]) + 1;
     * }
     */

    // Not self dependent due to different y (with offset).

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        y,
        0,
        10,
        Block::make({Store::make(
            a, {y + 1}, Add::make(Load::make(a, {y + 1}, 1), 1), 1)}));

    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[0] = (A[0]) + x;
     * }
     */

    // Is self dependent since all loops use a common constant element of A.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Block::make(
            {Store::make(a, {0}, Add::make(Load::make(a, {0}, 1), x), 1)}));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[0] = (B[0]) + x;
     * }
     */

    // Is not self dependent beacause there is no store to the buffer that is
    // read.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Block::make(
            {Store::make(a, {0}, Add::make(Load::make(b, {0}, 1), x), 1)}));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[y] = (A[y]) + x;
     * }
     */

    // Is self dependent since all loops use a common symbolic element of A.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Block::make(
            {Store::make(a, {y}, Add::make(Load::make(a, {y}, 1), x), 1)}));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x] = A[x + 1];
     * }
     */

    // In this case it depends if we are considering execution order.

    MemDependencyChecker analyzer;

    Stmt* stmt =
        For::make(x, 0, 10, Store::make(a, {x}, Load::make(a, {x + 1}, 1), 1));
    stmt->accept(&analyzer);

    // With analysis of order disabled, this is self dependent since the read
    // from X+1 and the write to X+1 could be in reverse order.
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x] = A[x + 1];
     * }
     */

    MemDependencyChecker analyzer;
    analyzer.allowLoopExecutionOrderAnalysis();

    Stmt* stmt =
        For::make(x, 0, 10, Store::make(a, {x}, Load::make(a, {x + 1}, 1), 1));
    stmt->accept(&analyzer);

    // If order analysis is enabled, this is not dependent since the read for
    // each element occurs before the write to that element.
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 1; x < 10; x++) {
     *   A[x] = A[x - 1];
     * }
     */

    MemDependencyChecker analyzer;

    Stmt* stmt =
        For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 1; x < 10; x++) {
     *   A[x] = A[x - 1];
     * }
     */

    MemDependencyChecker analyzer;
    analyzer.allowLoopExecutionOrderAnalysis();

    Stmt* stmt =
        For::make(x, 1, 10, Store::make(a, {x}, Load::make(a, {x - 1}, 1), 1));
    stmt->accept(&analyzer);

    // In this case, even with order analysis the Load is dependent on the
    // Store, since the write to X occurs before the read from X.
    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 9; x++) {
     *   A[9 - x] = A[8 - x];
     * }
     */

    // Still works if the execution order is reversed, so long as the read
    // comes before the write.

    MemDependencyChecker analyzer;
    analyzer.allowLoopExecutionOrderAnalysis();

    Stmt* stmt = For::make(
        x,
        3,
        10,
        Store::make(
            a, {ExprHandle(9) - x}, Load::make(a, {ExprHandle(8) - x}, 1), 1));
    stmt->accept(&analyzer);

    // However here was can determine the A store is earlier in the order than
    // the load.
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 9; x++) {
     *   A[8 - x] = A[9 - x];
     * }
     */

    // But not if it doesn't.

    MemDependencyChecker analyzer;
    analyzer.allowLoopExecutionOrderAnalysis();

    Stmt* stmt = For::make(
        x,
        3,
        10,
        Store::make(
            a, {ExprHandle(8) - x}, Load::make(a, {ExprHandle(9) - x}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 9; x++) {
     *   A[9 - x] = A[8 - x];
     * }
     */

    // And not if we're not relying on execution order.

    MemDependencyChecker analyzer;

    Stmt* stmt = For::make(
        x,
        3,
        10,
        Store::make(
            a, {ExprHandle(9) - x}, Load::make(a, {ExprHandle(8) - x}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 3; x < 10; x++) {
     *   A[x - 2] = A[x - 1];
     * }
     */

    // Forward order but negative indices.

    MemDependencyChecker analyzer;
    analyzer.allowLoopExecutionOrderAnalysis();

    Stmt* stmt = For::make(
        x, 3, 10, Store::make(a, {x - 2}, Load::make(a, {x - 1}, 1), 1));
    stmt->accept(&analyzer);

    // However here was can determine the A store is earlier in the order than
    // the load.
    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2];
     * }
     */

    // With an access stride.

    MemDependencyChecker analyzer;
    // Execution order doesn't matter since the read and the write are totally
    // distinct.

    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 1];
     * }
     */

    // Here we can use the common stride of the accesses to determine they are
    // distinct.
    // Note, this is the only place (loop self depedency) we use this stride
    // to avoid unnecessary depedence.

    MemDependencyChecker analyzer;
    // Execution order doesn't matter since the read and the write are totally
    // distinct.

    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 1}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 - 1];
     * }
     */

    // same if the read is behind the write so long as they are distinct.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 1, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 - 1}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 2];
     * }
     */

    // But not if the offset is in the stride.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 2}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 - 2];
     * }
     */

    // Works with negative offsets too.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 1, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 - 2}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 7];
     * }
     */

    // Detects accesses are distinct when offset is large but not a multiple
    // of stride.
    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 7}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 2 + 4];
     * }
     */

    // Works with offsets which are multiples of the stride.
    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 2 + 4}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 6] = A[x * 6 + 5];
     * }
     */

    // detects accesses are distinct with large strides when the offset is
    // within.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 6}, Load::make(a, {x * 6 + 5}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 6];
     * }
     */

    // detects accesses are overlapping when stride is different but a
    // multiple.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 6}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 4] = A[x * 2];
     * }
     */

    // still works when the read axis is the smaller stride.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 4}, Load::make(a, {x * 2}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 6 + 1];
     * }
     */

    // detects accesses are distinct when stride is different but a multiple
    // and there is an offset.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 6 + 1}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 6 + 4];
     * }
     */

    // The smaller stride determines whether there is overlap.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 6 + 4}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2 + 3] = A[x * 6];
     * }
     */

    // The smaller stride determines whether there is overlap, not the larger.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2 + 3}, Load::make(a, {x * 6}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[x * 3 + 1];
     * }
     */

    // If they have strides with no common muliple > 1, they overlap.
    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x * 2}, Load::make(a, {x * 3 + 1}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x] = A[x + 10];
     * }
     */

    // If the offset is greater than the size of the loop, they can't overlap.

    MemDependencyChecker analyzer;
    Stmt* stmt =
        For::make(x, 0, 10, Store::make(a, {x}, Load::make(a, {x + 10}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x] = A[9 - x];
     * }
     */

    // If they have different execution orders they may overlap.
    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Store::make(a, {x}, Load::make(a, {ExprHandle(9) - x}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x * 2] = A[19 - x * 2];
     * }
     */

    // Or they may not, depending on their start offset and strides.
    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Store::make(a, {x * 2}, Load::make(a, {ExprHandle(19) - x * 2}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x / 2] = A[x / 2];
     * }
     */

    // If the stride is not monotonic, they overlap.

    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x / 2}, Load::make(a, {x / 2}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x / 2] = A[x / 2] + 1;
     * }
     */

    // If the stride is not monotonic, they overlap - even with an offset.
    MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x, 0, 10, Store::make(a, {x / 2}, Load::make(a, {x / 2 + 1}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   A[x % 2] = A[x % 2];
     * }
     */

    // Mod too...

    analysis::MemDependencyChecker analyzer;
    Stmt* stmt = For::make(
        x,
        0,
        10,
        Store::make(
            a, {Mod::make(x, 2)}, Load::make(a, {Mod::make(x, 2)}, 1), 1));
    stmt->accept(&analyzer);

    ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
  }

  {
    /* for (int x = y; x < z; x++) {
     *   A[x] = A[x + 1];
     * }
     */

    // Still works with symbolic loop extents.

    {
      MemDependencyChecker analyzer;
      Stmt* stmt =
          For::make(x, y, z, Store::make(a, {x}, Load::make(a, {x + 1}, 1), 1));
      stmt->accept(&analyzer);

      ASSERT_TRUE(isSelfDependent(analyzer.getHistory()));
    }

    {
      MemDependencyChecker analyzer;
      analyzer.allowLoopExecutionOrderAnalysis();
      Stmt* stmt =
          For::make(x, y, z, Store::make(a, {x}, Load::make(a, {x + 1}, 1), 1));
      stmt->accept(&analyzer);

      ASSERT_FALSE(isSelfDependent(analyzer.getHistory()));
    }
  }
}

// Verify that a strided access still works.
// TODO: actually this only works because of the size of the ranges, revist this
// test after strided overlap is implemented.
TEST(MemDependency, MemDependencyCheckerLoopDistinctStrides) {
  KernelScope kernel_scope;
  BufHandle a("A", {20}, kInt);
  BufHandle b("B", {20}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  using namespace analysis;
  MemDependencyChecker analyzer({a.node()}, {b.node()});
  Stmt* stmt = Block::make(
      {For::make(
           x,
           0,
           10,
           Store::make(b, {x * 2 + 1}, Load::make(a, {x * 2 + 1}, 1), 1)),
       For::make(
           x, 0, 10, Store::make(b, {x * 2}, Load::make(a, {x * 2}, 1), 1))

      });
  stmt->accept(&analyzer);

  // Sanity check output depends on input.
  ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

  // Output has 2 dependencies... the store in each loop.
  auto outputAccess = analyzer.output(b.node());
  ASSERT_EQ(outputAccess->dependencies().size(), 2);
}

/* TODO(nickg) - this test will fail due to the lack of stride math in Bound
TEST(MemDependency, MemDependencyCheckerLoopDistinctStrides) {
  KernelScope kernel_scope;
  BufHandle a("A", {20}, kInt);
  BufHandle b("B", {20}, kInt);
  BufHandle c("C", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    analysis::MemDependencyChecker analyzer({a.node()}, {c.node()});
    Stmt* stmt = Block::make(
        {For::make(
             x,
             0,
             10,
             Store::make(b, {x * 2 + 1}, Load::make(a, {x * 2 + 1}, 1), 1)),
         For::make(
             x, 0, 10, Store::make(b, {x * 2}, Load::make(a, {x * 2}, 1), 1)),
         For::make(x, 0, 10, Store::make(c, {x}, Load::make(b, {x}, 1), 1))

        });
    stmt->accept(&analyzer);

    std::cout << *stmt << "\n";
    for (auto& wi : analyzer.getHistory()) {
      wi->print();
    }
  }
}*/

// analysis on Stmts using Cond.
TEST(MemDependency, MemDependencyCheckerLoopBoundsCond) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  BufHandle c("C", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  using namespace analysis;

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * if (y<5 ? 1 : 0) {
     *   C[0] = (B[0]) + 1;
     * } else {
     *   C[0] = (B[1]) + 1;
     * }
     */

    // Future usages may depend on accesses in both branches of a condition.

    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}, 1), 1)),
         Cond::make(
             CompareSelect::make(y, 5, CompareSelectOperation::kLT),
             Store::make(c, {0}, Add::make(Load::make(b, {0}, 1), 1), 1),
             Store::make(c, {0}, Add::make(Load::make(b, {1}, 1), 1), 1))});

    stmt->accept(&analyzer);

    // Output C should have 3 dependencies, each of the three stores.
    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 3);

    // C depends indirectly on A and B.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * if (y<5 ? 1 : 0) {
     *   for (int x = 0; x < 10; x++) {
     *     C[x] = B[x];
     *   }
     * } else {
     *   for (int x = 0; x < 10; x++) {
     *     C[x] = (B[x]) + 1;
     *   }
     * }
     */

    // Future usages may depend on accesses in both branches of a condition.

    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}, 1), 1)),
         Cond::make(
             CompareSelect::make(y, 5, CompareSelectOperation::kLT),
             For::make(x, 0, 10, Store::make(c, {x}, Load::make(b, {x}, 1), 1)),
             For::make(
                 x,
                 0,
                 10,
                 Store::make(
                     c, {x}, Add::make(Load::make(b, {x}, 1), 1), 1)))});

    stmt->accept(&analyzer);

    // Output C should have 3 dependencies, each of the three stores.
    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 3);

    // TODO(nickg): actually since the true and false branch cover the total
    // range of the first store this should have 2 dependencies, but we don't
    // do that yet.

    // C depends indirectly on A and B.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * if (y<5 ? 1 : 0) {
     *   for (int x = 0; x < 10; x++) {
     *     C[x] = (B[x]) + 1;
     *   }
     * }
     */

    // Only has true branch.

    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}, 1), 1)),
         Cond::make(
             CompareSelect::make(y, 5, CompareSelectOperation::kLT),
             For::make(
                 x,
                 0,
                 10,
                 Store::make(c, {x}, Add::make(Load::make(b, {x}, 1), 1), 1)),
             nullptr)});

    stmt->accept(&analyzer);

    // Output C should have 3 dependencies, each of the three stores.
    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 2);

    // C depends indirectly on A and B.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * if (y<5 ? 1 : 0) {
     * } else {
     *   for (int x = 0; x < 10; x++) {
     *     C[x] = (B[x]) + 1;
     *   }
     * }
     */

    // Only has false branch.

    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}, 1), 1)),
         Cond::make(
             CompareSelect::make(y, 5, CompareSelectOperation::kLT),
             nullptr,
             For::make(
                 x,
                 0,
                 10,
                 Store::make(
                     c, {x}, Add::make(Load::make(b, {x}, 1), 1), 1)))});

    stmt->accept(&analyzer);

    // Output C should have 3 dependencies, each of the three stores.
    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 2);

    // C depends indirectly on A and B.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * if (C[0]<5 ? 1 : 0) {
     *   C[0] = 5;
     * }
     */

    // Cond's Condition depends on a previous access.

    MemDependencyChecker analyzer({a}, {c});
    Store* initStore = Store::make(c, {x}, Load::make(a, {x}, 1), 1);
    ExprHandle conditionalLoad = Load::make(c, {0}, 1);
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, initStore),
         Cond::make(
             CompareSelect::make(
                 conditionalLoad, 5, CompareSelectOperation::kLT),
             Store::make(c, {0}, 5, 1),
             nullptr)});

    stmt->accept(&analyzer);

    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));

    ASSERT_TRUE(analyzer.dependsDirectly(conditionalLoad.node(), initStore));
    ASSERT_FALSE(analyzer.dependsDirectly(conditionalLoad.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(conditionalLoad.node(), a.node()));
  }
}

// Stmts using IfThenElse.
TEST(MemDependency, MemDependencyCheckerIfThenElse) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  BufHandle c("C", {10}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  using namespace analysis;

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * C[0] = (y < 5 ? (B[0]) + 1 : (B[1]) + 1;
     */

    // Future usages may depend on accesses in both branches of a condition.

    MemDependencyChecker analyzer({a, b}, {c});
    Store* ifStore = Store::make(
        c,
        {0},
        IfThenElse::make(
            CompareSelect::make(y, 5, CompareSelectOperation::kLT),
            Add::make(Load::make(b, {0}, 1), 1),
            Add::make(Load::make(b, {1}, 1), 1)),
        1);
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}, 1), 1)),
         ifStore});

    stmt->accept(&analyzer);

    // Output C should have 2 dependencies, each of the two stores.
    auto outputAccess = analyzer.output(c.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 2);

    // Now we need to check the Store containing the IfThenElse.
    auto ifStoreAccess = analyzer.accessFor(ifStore);

    // It should have 2 dependencies.
    ASSERT_EQ(ifStoreAccess->dependencies().size(), 2);

    // C depends indirectly on A and B.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[x];
     * }
     * C[0] = (y < 5 ? (B[0]) + 1 : 42;
     */

    // If the load appears in only one side of an IfThenElse the output may be
    // dependent on it.

    MemDependencyChecker analyzer({a, b}, {c});
    Store* ifStore = Store::make(
        c,
        {0},
        IfThenElse::make(
            CompareSelect::make(y, 5, CompareSelectOperation::kLT),
            Add::make(Load::make(b, {0}, 1), 1),
            42),
        1);
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, Store::make(c, {x}, Load::make(a, {x}, 1), 1)),
         ifStore});

    stmt->accept(&analyzer);

    // C depends indirectly on A and B.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = (x < 5 ? B[x] : A[x];
     * }
     */

    // In this case C is dependent on both A and B.

    // TODO: in cases like this it would be possible to split the range of B
    // into two bounds, one dependent on A and one depenent on B. We'd need to
    // examine conditions relative to previously encountered loop variables. I'm
    // uncertain if this would be helpful.

    MemDependencyChecker analyzer({a, b}, {c});
    Store* ifStore = Store::make(
        c,
        {0},
        IfThenElse::make(
            CompareSelect::make(y, 5, CompareSelectOperation::kLT),
            Load::make(b, {x}, 1),
            Load::make(a, {x}, 1)),
        1);
    Stmt* stmt = Block::make({For::make(x, 0, 10, ifStore)});

    stmt->accept(&analyzer);

    // C depends indirectly on A and B.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));
  }
}

// Cutting a loop with single elem writes
TEST(MemDependency, MemDependencyCheckerCutLoop) {
  KernelScope kernel_scope;
  BufHandle a("A", {10}, kInt);
  BufHandle b("B", {10}, kInt);
  VarHandle x("x", kInt);

  using namespace analysis;

  {
    /* for (int x = 0; x < 10; x++) {
     *   B[x] = A[x];
     * }
     * B[5] = 100;
     */

    // Cutting a loop with single element writes.

    MemDependencyChecker analyzer({a}, {b});
    Stmt* stmt = Block::make(
        {For::make(x, 0, 10, Store::make(b, {x}, Load::make(a, {x}, 1), 1)),
         Store::make(b, {5}, 100, 1)});

    stmt->accept(&analyzer);

    // Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // Output has 2 depdenencies.
    auto outputAccess = analyzer.output(b.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 2);
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   B[x] = A[x];
     * }
     * for (int x = 4; x < 7; x++) {
     *   B[x] = B[x] + 3;
     * }
     * B[5] = 100;
     * B[6] = 101;
     * B[7] = 102;
     */

    // Cutting a loop with a smaller loop but then totally overlap that second
    // loop with one element writes.

    MemDependencyChecker analyzer({a}, {b});
    For* firstLoop =
        For::make(x, 0, 10, Store::make(b, {x}, Load::make(a, {x}, 1), 1));
    Store* secondStore =
        Store::make(b, {x}, Add::make(Load::make(b, {x}, 1), 1), 3);
    For* secondLoop = For::make(x, 4, 7, secondStore);

    Stmt* stmt = Block::make(
        {firstLoop,
         secondLoop,
         Store::make(b, {4}, 100, 1),
         Store::make(b, {5}, 101, 1),
         Store::make(b, {6}, 102, 1)});

    stmt->accept(&analyzer);

    // Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // Output has 4 depdenencies.
    auto outputAccess = analyzer.output(b.node());
    ASSERT_NE(outputAccess, nullptr);
    ASSERT_EQ(outputAccess->dependencies().size(), 4);

    // Second loop depends on first loop.
    ASSERT_TRUE(analyzer.dependsDirectly(secondLoop, firstLoop));

    // Output does not depend on second loop or store.
    ASSERT_FALSE(analyzer.dependsIndirectly(b.node(), secondLoop));
    ASSERT_FALSE(analyzer.dependsIndirectly(b.node(), secondStore));
  }
}

// Dynamic shapes (load in indices).
TEST(MemDependency, MemDependencyCheckerDynamicShapes) {
  KernelScope kernel_scope;
  BufHandle a("A", {100}, kInt);
  BufHandle b("B", {100}, kInt);
  BufHandle c("C", {100}, kInt);
  VarHandle x("x", kInt);

  using namespace analysis;

  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };

  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  {
    /* for (int x = 0; x < B[0]; x++) {
     *   C[x] = A[x];
     * }
     */
    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        Load::make(b, {0}, 1),
        Store::make(c, {x}, Load::make(a, {x}, 1), 1))});

    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 2
     *  1. Input: A[(0, 99)] - dependents: 3
     *  2. Load: B[(0, 0)] - depends on: 0  - dependents: 3 4
     *  3. Load: A[(0, (B[0]) - 1)] - depends on: 1 2  - dependents: 4
     *  4. Store: C[(0, (B[0]) - 1)] - depends on: 2 3  - dependents: 5
     *  5. Output: C[(0, 99)] - depends on: 4
     */

    // Output dependent on A input.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    // Also dependent on B input to determine the size of the region written.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // The accesses in the loop depend on the load in the stop condition.
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[3]->hasDependency(history[2]));

    // Make a load from B to compare against.
    ExprHandle loadFromB = Load::make(b, {0}, 1);

    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, loadFromB - 1)}));
    ASSERT_TRUE(EQ(history[4]->bounds(), {CB(0, loadFromB - 1)}));
  }

  {
    /* for (int x = B[0]; x < B[1]; x++) {
     *   C[x] = A[x];
     * }
     */
    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make({For::make(
        x,
        Load::make(b, {0}, 1),
        Load::make(b, {1}, 1),
        Store::make(c, {x}, Load::make(a, {x}, 1), 1))});

    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 2 3
     *  1. Input: A[(0, 99)] - dependents: 4
     *  2. Load: B[(0, 0)] - depends on: 0  - dependents: 4 5
     *  3. Load: B[(1, 1)] - depends on: 0  - dependents: 4 5
     *  4. Load: A[(B[0], (B[1]) - 1)] - depends on: 1 2 3  - dependents: 5
     *  5. Store: C[(B[0], (B[1]) - 1)] - depends on: 2 3 4  - dependents: 6
     *  6. Output: C[(0, 99)] - depends on: 5
     */

    // Sanity check output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 7);

    // The accesses in the loop depend on the load in the start condition.
    ASSERT_TRUE(history[5]->hasDependency(history[2]));
    ASSERT_TRUE(history[4]->hasDependency(history[2]));

    // also the stop condition.
    ASSERT_TRUE(history[5]->hasDependency(history[3]));
    ASSERT_TRUE(history[4]->hasDependency(history[3]));

    // Make loads from B to compare against.
    ExprHandle loadFromB0 = Load::make(b, {0}, 1);
    ExprHandle loadFromB1 = Load::make(b, {1}, 1);
    ASSERT_TRUE(EQ(history[4]->bounds(), {CB(loadFromB0, loadFromB1 - 1)}));
    ASSERT_TRUE(EQ(history[5]->bounds(), {CB(loadFromB0, loadFromB1 - 1)}));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[x] = A[B[x]];
     * }
     */
    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        10,
        Store::make(c, {x}, Load::make(a, {Load::make(b, {x}, 1)}, 1), 1))});

    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 2
     *  1. Input: A[(0, 99)] - dependents: 3
     *  2. Load: B[(0, 9)] - depends on: 0  - dependents: 3 4
     *  3. Load: A[(B[0], B[9])] - depends on: 1 2  - dependents: 4
     *  4. Store: C[(0, 9)] - depends on: 2 3  - dependents: 5
     *  5. Output: C[(0, 99)] - depends on: 4
     */

    // Sanity check output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // The store depends on both loads, the load of A depends on the load of B.
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[4]->hasDependency(history[3]));

    ASSERT_TRUE(history[3]->hasDependency(history[2]));

    // The loads in the indices depend on the relevant input buffer.
    ASSERT_TRUE(history[3]->hasDependency(history[1]));
    ASSERT_TRUE(history[2]->hasDependency(history[0]));

    // The load from B has the loop bounds.
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 9)}));

    // The load from A has bounds B[0] to B[9].
    ExprHandle loadFromB0 = Load::make(b, {0}, 1);
    ExprHandle loadFromB9 = Load::make(b, {9}, 1);
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(loadFromB0, loadFromB9)}));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[B[x]] = A[x];
     * }
     */
    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        10,
        Store::make(c, {Load::make(b, {x}, 1)}, Load::make(a, {x}, 1), 1))});

    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 3
     *  1. Input: A[(0, 99)] - dependents: 2
     *  2. Load: A[(0, 9)] - depends on: 1  - dependents: 4
     *  3. Load: B[(0, 9)] - depends on: 0  - dependents: 4
     *  4. Store: C[(B[0], B[9])] - depends on: 2 3  - dependents: 5
     *  5. Output: C[(0, 99)] - depends on: 4
     */
    // Sanity check output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // The store depends on both loads, neither load is dependent.
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[4]->hasDependency(history[3]));

    ASSERT_FALSE(history[3]->hasDependency(history[2]));
    ASSERT_FALSE(history[2]->hasDependency(history[3]));

    // The loads each depend on their relevant input. (but accesses are in a
    // different order than the last case).
    ASSERT_TRUE(history[3]->hasDependency(history[0]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));

    // The load from B has the loop bounds.
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, 9)}));

    // And so does the load from A.
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 9)}));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   C[B[A[x]]] = x;
     * }
     */
    MemDependencyChecker analyzer({a, b}, {c});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        10,
        Store::make(c, {Load::make(b, {Load::make(a, {x}, 1)}, 1)}, x, 1))});

    stmt->accept(&analyzer);

    /*  0. Input: B[(0, 99)] - dependents: 3
     *  1. Input: A[(0, 99)] - dependents: 2
     *  2. Load: A[(0, 9)] - depends on: 1  - dependents: 3 4
     *  3. Load: B[(A[0], A[9])] - depends on: 0 2  - dependents: 4
     *  4. Store: C[(B[A[0]], B[A[9]])] - depends on: 2 3  - dependents: 5
     *  5. Output: C[(0, 99)] - depends on: 4
     */

    // Sanity check output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(c.node(), b.node()));

    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // The store depends on both loads.
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[4]->hasDependency(history[3]));

    // The outer load depends on the inner.
    ASSERT_TRUE(history[3]->hasDependency(history[2]));

    // The loads each depend on their relevant input. (but accesses are in a
    // different order than the last case).
    ASSERT_TRUE(history[3]->hasDependency(history[0]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));

    // The load from A has the loop bounds.
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 9)}));
    // The load from B as bounds A[0] to A[9].
    ExprHandle loadFromA0 = Load::make(a, {0}, 1);
    ExprHandle loadFromA9 = Load::make(a, {9}, 1);
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(loadFromA0, loadFromA9)}));

    // The store has bounds of B[A[0]] to B[A[9]].
    ExprHandle loadFromBA0 = Load::make(b, {loadFromA0}, 1);
    ExprHandle loadFromBA9 = Load::make(b, {loadFromA9}, 1);
    ASSERT_TRUE(EQ(history[4]->bounds(), {CB(loadFromBA0, loadFromBA9)}));
  }
}

// Verify multi dimensional bounds work.
TEST(MemDependency, MemDependencyCheckerMultiDim) {
  KernelScope kernel_scope;
  int M = 10, N = 9, K = 12;
  BufHandle a("A", {M, N, K}, kInt);
  BufHandle b("B", {M, N, K}, kInt);
  BufHandle c("C", {M, K}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  using namespace analysis;

  auto CB = [](ExprHandle s, ExprHandle e) {
    return Bound(s.node(), e.node());
  };

  auto EQ = [](const IndexBounds& x, const IndexBounds& y) {
    return indexBoundsEquals(x, y);
  };

  {
    /* for (int x = 0; x < 10; x++) {
     *   for (int y = 0; y < 9; y++) {
     *     for (int z = 0; z < 12; z++) {
     *       B[x, y, z] = A[x, y, z];
     *     }
     *   }
     * }
     */
    // Full range.

    MemDependencyChecker analyzer({a}, {b});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        M,
        For::make(
            y,
            0,
            N,
            For::make(
                z,
                0,
                K,
                Store::make(b, {x, y, z}, Load::make(a, {x, y, z}, 1), 1))))});

    stmt->accept(&analyzer);

    // Sanity test: Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // 4 accesses: input, load, store, output.
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 4);

    // Simple chain from input to output.
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    ASSERT_TRUE(history[1]->hasDependency(history[0]));

    ASSERT_TRUE(
        EQ(history[1]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
  }

  {
    /* for (int x = 0; x < 5; x++) {
     *   for (int y = 0; y < 5; y++) {
     *     for (int z = 0; z < 5; z++) {
     *       B[x, y, z] = A[x, y, z];
     *     }
     *   }
     * }
     */
    // Partial range.

    MemDependencyChecker analyzer({a}, {b});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        5,
        For::make(
            y,
            0,
            5,
            For::make(
                z,
                0,
                5,
                Store::make(b, {x, y, z}, Load::make(a, {x, y, z}, 1), 1))))});

    stmt->accept(&analyzer);

    // Sanity test: Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // 4 accesses: input, load, store, output.
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 4);

    // Simple chain from input to output.
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    ASSERT_TRUE(history[1]->hasDependency(history[0]));

    ASSERT_TRUE(EQ(history[1]->bounds(), {CB(0, 4), CB(0, 4), CB(0, 4)}));
    ASSERT_TRUE(EQ(history[2]->bounds(), {CB(0, 4), CB(0, 4), CB(0, 4)}));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   for (int y = 0; y < 12; y++) {
     *     B[x, 0, y] = A[x, 0, y];
     *   }
     * }
     */

    // Partial loops.

    MemDependencyChecker analyzer({a}, {b});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        N,
        For::make(
            y,
            0,
            K,
            Store::make(b, {x, 0, y}, Load::make(a, {x, 0, y}, 1), 1)))});

    stmt->accept(&analyzer);

    // Sanity test: Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // 4 accesses: input, load, store, output.
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 4);

    // Simple chain from input to output.
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    ASSERT_TRUE(history[1]->hasDependency(history[0]));

    ASSERT_TRUE(
        EQ(history[1]->bounds(), {CB(0, N - 1), CB(0, 0), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, N - 1), CB(0, 0), CB(0, K - 1)}));
  }

  {
    /* for (int x = 0; x < 10; x++) {
     *   for (int y = 0; y < 100; y++) {
     *     for (int z = 0; z < 12; z++) {
     *       B[x, 0, z] = (A[x, 0, z]) + (C[x, z]);
     *     }
     *   }
     * }
     */

    // Loops that don't correspond to an index, bufs with different
    // dimensionality.

    MemDependencyChecker analyzer({a, c}, {b});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        M,
        For::make(
            y,
            0,
            100,
            For::make(
                z,
                0,
                K,
                Store::make(
                    b,
                    {x, 0, z},
                    Add::make(
                        Load::make(a, {x, 0, z}, 1), Load::make(c, {x, z}, 1)),
                    1))))});

    stmt->accept(&analyzer);

    // Sanity test: Output depends on both inputs.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), c.node()));

    // 6 accesses: 2 inputs, 2 loads, store, output.
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 6);

    // Simple chain from input to output over the A buf.
    // history[0] is the C input, history[3] is the load from C.
    ASSERT_TRUE(history[5]->hasDependency(history[4]));
    ASSERT_TRUE(history[4]->hasDependency(history[2]));
    ASSERT_TRUE(history[2]->hasDependency(history[1]));
    // The store also depends on the load from the C input.
    ASSERT_TRUE(history[4]->hasDependency(history[3]));
    ASSERT_TRUE(history[3]->hasDependency(history[0]));

    // A Buf accesses.
    ASSERT_TRUE(
        EQ(history[4]->bounds(), {CB(0, M - 1), CB(0, 0), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, M - 1), CB(0, 0), CB(0, K - 1)}));

    // C buf access.
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, M - 1), CB(0, K - 1)}));
  }

  {
    /* for (int x = 0; x < 9; x++) {
     *   for (int y = 0; y < 10; y++) {
     *     for (int z = 0; z < 12; z++) {
     *       B[x, 0, 0] = (B[x, y, z]) + (A[x, y, z]);
     *     }
     *   }
     * }
     */
    // Multi-dim reductions.

    MemDependencyChecker analyzer({a}, {b});
    Stmt* stmt = Block::make({For::make(
        x,
        0,
        M,
        For::make(
            y,
            0,
            N,
            For::make(
                z,
                0,
                K,
                Store::make(
                    b,
                    {x, 0, 0},
                    Add::make(
                        Load::make(b, {x, y, z}, 1),
                        Load::make(a, {x, y, z}, 1)),
                    1))))});

    stmt->accept(&analyzer);

    // Sanity test: Output depends on input.
    ASSERT_TRUE(analyzer.dependsIndirectly(b.node(), a.node()));

    // 4 accesses: input, 2 loads, store, output.
    auto history = analyzer.getHistory();
    ASSERT_EQ(history.size(), 5);

    // Simple chain from input to output.
    ASSERT_TRUE(history[4]->hasDependency(history[3]));
    ASSERT_TRUE(history[3]->hasDependency(history[2]));
    ASSERT_TRUE(history[3]->hasDependency(history[1]));
    ASSERT_TRUE(history[2]->hasDependency(history[0]));

    // The load from B depends on the store to B.
    ASSERT_TRUE(history[1]->hasDependency(history[3]));

    ASSERT_TRUE(
        EQ(history[1]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
    ASSERT_TRUE(
        EQ(history[2]->bounds(), {CB(0, M - 1), CB(0, N - 1), CB(0, K - 1)}));
    ASSERT_TRUE(EQ(history[3]->bounds(), {CB(0, M - 1), CB(0, 0), CB(0, 0)}));
  }
}

// Various tests using the external Compute/Reduce API.
TEST(MemDependency, MemDependencyCheckerComputeAPI) {
  KernelScope kernel_scope;

  using namespace analysis;

  /* for (int m = 0; m < 4; m++) {
   *   for (int n = 0; n < 5; n++) {
   *     for (int k = 0; k < 6; k++) {
   *       broadcast_add[m, n, k] = (a[m, n]) + (b[n, k]);
   *     }
   *   }
   * }
   * for (int m_1 = 0; m_1 < 4; m_1++) {
   *   for (int n_1 = 0; n_1 < 5; n_1++) {
   *     for (int k_1 = 0; k_1 < 6; k_1++) {
   *       d[m_1, n_1, k_1] = (broadcast_add(m_1, n_1, k_1)) + float(1);
   *     }
   *   }
   * }
   */

  // Can determine if 2 loops created by Compute are dependent.
  Placeholder a_buf("a", kFloat, {4, 5});
  Placeholder b_buf("b", kFloat, {5, 6});
  Tensor* c = Compute(
      "broadcast_add",
      {{4, "m"}, {5, "n"}, {6, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  Tensor* d = Compute(
      "d",
      {{4, "m"}, {5, "n"}, {6, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c->call(m, n, k) + 1;
      });

  LoopNest l({d}, {c, d});

  MemDependencyChecker analyzer({a_buf.data(), b_buf.data()}, {d->buf()});

  l.root_stmt()->accept(&analyzer);

  // Sanity test: Output depends on input.
  ASSERT_TRUE(analyzer.dependsIndirectly(d->buf(), a_buf.data()));
  ASSERT_TRUE(analyzer.dependsIndirectly(d->buf(), b_buf.data()));

  // Second loop depends on first loop.
  auto* c_loop = l.getLoopStmtsFor(c)[0];
  auto* d_loop = l.getLoopStmtsFor(d)[0];
  ASSERT_TRUE(analyzer.dependsDirectly(d_loop, c_loop));
}

TEST(MemDependency, MemDependencyCheckerComputeInline) {
  KernelScope kernel_scope;

  using namespace analysis;

  /* for (int m = 0; m < 4; m++) {
   *   for (int n = 0; n < 5; n++) {
   *     for (int k = 0; k < 6; k++) {
   *       d[m, n, k] = ((a[m, n]) + (b[n, k])) + float(1);
   *     }
   *   }
   * }
   */

  // Check inlining affects the number of accesses returned.

  Placeholder a_buf("a", kFloat, {4, 5});
  Placeholder b_buf("b", kFloat, {5, 6});
  Tensor* c = Compute(
      "broadcast_add",
      {{4, "m"}, {5, "n"}, {6, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });
  Tensor* d = Compute(
      "d",
      {{4, "m"}, {5, "n"}, {6, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return c->call(m, n, k) + 1;
      });

  LoopNest l({d}, {c, d});
  l.computeInline(c->buf());

  MemDependencyChecker analyzer({a_buf.data(), b_buf.data()}, {d->buf()});
  l.root_stmt()->accept(&analyzer);

  // Sanity test: Output depends on input.
  ASSERT_TRUE(analyzer.dependsIndirectly(d->buf(), a_buf.data()));
  ASSERT_TRUE(analyzer.dependsIndirectly(d->buf(), b_buf.data()));

  // broadcast_add tensor should not appear in trace at all.
  for (auto& wi : analyzer.getHistory()) {
    ASSERT_NE(wi->var(), c->buf()->base_handle());
  }
}

TEST(MemDependency, MemDependencyCheckerComputeSplit) {
  KernelScope kernel_scope;

  using namespace analysis;
  // Split an axis, so the number of loops != the number of dimensions.

  Placeholder a_buf("a", kFloat, {4, 5});
  Placeholder b_buf("b", kFloat, {5, 6});
  Tensor* c = Compute(
      "broadcast_add",
      {{4, "m"}, {5, "n"}, {6, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });

  LoopNest l({c});

  MemDependencyChecker analyzer_before(
      {a_buf.data(), b_buf.data()}, {c->buf()});
  l.root_stmt()->accept(&analyzer_before);

  For *o, *i, *t;
  l.splitWithTail(l.getLoopStmtsFor(c)[0], 2, &o, &i, &t);

  MemDependencyChecker analyzer_after({a_buf.data(), b_buf.data()}, {c->buf()});
  Stmt* stmt = IRSimplifier::simplify(l.root_stmt());
  stmt->accept(&analyzer_after);

  // Splitting should not change accesses at all.
  auto history_before = analyzer_before.getHistory();
  auto history_after = analyzer_after.getHistory();

  ASSERT_EQ(history_before.size(), history_after.size());

  for (size_t i = 0; i < history_before.size(); ++i) {
    ASSERT_EQ(history_before[i]->type(), history_after[i]->type());
    ASSERT_EQ(history_before[i]->var(), history_after[i]->var());
    ASSERT_EQ(
        history_before[i]->bounds().size(), history_after[i]->bounds().size());
    ASSERT_TRUE(indexBoundsEquals(
        history_before[i]->bounds(), history_after[i]->bounds()));
    ASSERT_EQ(
        history_before[i]->dependencies().size(),
        history_after[i]->dependencies().size());
    ASSERT_EQ(
        history_before[i]->dependents().size(),
        history_after[i]->dependents().size());
  }
}

TEST(MemDependency, MemDependencyCheckerComputeReorder) {
  KernelScope kernel_scope;

  using namespace analysis;
  // Reorder an axis, so the loop order doesn't match the indexing order.

  Placeholder a_buf("a", kFloat, {4, 5});
  Placeholder b_buf("b", kFloat, {5, 6});
  Tensor* c = Compute(
      "broadcast_add",
      {{4, "m"}, {5, "n"}, {6, "k"}},
      [&](const VarHandle& m, const VarHandle& n, const VarHandle& k) {
        return a_buf.load(m, n) + b_buf.load(n, k);
      });

  LoopNest l({c});

  MemDependencyChecker analyzer_before(
      {a_buf.data(), b_buf.data()}, {c->buf()});
  l.root_stmt()->accept(&analyzer_before);

  auto loops = l.getLoopStmtsFor(c);
  l.reorderAxis(loops[0], loops[1]);

  MemDependencyChecker analyzer_after({a_buf.data(), b_buf.data()}, {c->buf()});
  Stmt* stmt = IRSimplifier::simplify(l.root_stmt());
  stmt->accept(&analyzer_after);

  // Reordering should not change accesses at all.
  auto history_before = analyzer_before.getHistory();
  auto history_after = analyzer_after.getHistory();

  ASSERT_EQ(history_before.size(), history_after.size());

  for (size_t i = 0; i < history_before.size(); ++i) {
    ASSERT_EQ(history_before[i]->type(), history_after[i]->type());
    ASSERT_EQ(history_before[i]->var(), history_after[i]->var());
    ASSERT_EQ(
        history_before[i]->bounds().size(), history_after[i]->bounds().size());
    ASSERT_TRUE(indexBoundsEquals(
        history_before[i]->bounds(), history_after[i]->bounds()));
    ASSERT_EQ(
        history_before[i]->dependencies().size(),
        history_after[i]->dependencies().size());
    ASSERT_EQ(
        history_before[i]->dependents().size(),
        history_after[i]->dependents().size());
  }
}

TEST(MemDependency, MemDependencyCheckerComputeReduce) {
  KernelScope kernel_scope;

  using namespace analysis;
  /* for (int l2 = 0; l2 < 2; l2++) {
   *   for (int n1 = 0; n1 < 3; n1++) {
   *     for (int m1 = 0; m1 < 6; m1++) {
   *       scale[l2, n1, m1] = (b[l2, n1, m1]) * (a[l2, n1, m1]);
   *     }
   *   }
   * }
   * for (int l1 = 0; l1 < 2; l1++) {
   *   sum[l1] = float(0);
   *   for (int n1_1 = 0; n1_1 < 3; n1_1++) {
   *     for (int m1_1 = 0; m1_1 < 6; m1_1++) {
   *       sum[l1] = ReduceOp(sum, (sum[l1]) + (scale(l1, n1_1, m1_1)),
   *                    out_args={l1}, reduce_args={n1, m1});
   *     }
   *   }
   * }
   */

  // Can determine dependencies of a Reduction.

  Placeholder a(BufHandle("a", {2, 3, 6}, kFloat));
  Placeholder b(BufHandle("b", {2, 3, 6}, kFloat));

  Tensor* c = Compute(
      "scale",
      {{2, "l2"}, {3, "n1"}, {6, "m1"}},
      [&](const VarHandle& l, const VarHandle& n, const VarHandle& m) {
        return b.load(l, n, m) * a.load(l, n, m);
      });
  Tensor* d = Reduce("sum", {{2, "l1"}}, Sum(), c, {{3, "n1"}, {6, "m1"}});
  LoopNest l({d}, {c, d});

  MemDependencyChecker analyzer({a.data(), b.data()}, {d->buf()});

  l.root_stmt()->accept(&analyzer);

  // Sanity test: Output depends on input.
  ASSERT_TRUE(analyzer.dependsIndirectly(d->buf(), a.data()));
  ASSERT_TRUE(analyzer.dependsIndirectly(d->buf(), b.data()));

  // Second loop depends on first loop.
  auto* c_loop = l.getLoopStmtsFor(c)[0];
  auto* d_loop = l.getLoopStmtsFor(d)[0];
  ASSERT_TRUE(analyzer.dependsDirectly(d_loop, c_loop));

  // Reduction depends on both inputs.
  auto reduces = NodeFinder<ReduceOp>::find(l.root_stmt());
  ASSERT_TRUE(analyzer.dependsIndirectly(reduces[0], a.data()));
  ASSERT_TRUE(analyzer.dependsIndirectly(reduces[0], b.data()));
}

TEST(MemDependency, MemDependencyCheckerComputeGEMM) {
  KernelScope kernel_scope;
  int M = 1024;
  int N = 1024;
  int K = 2048;
  using namespace analysis;

  Placeholder AP(BufHandle("A", {M, K}, kFloat));
  Placeholder BP(BufHandle("B", {K, N}, kFloat));
  Tensor* CT = Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* m = loops[0];
    For* mo;
    For* mi;
    loop.splitWithMask(m, 4, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* n = loops[2];
    For* no;
    For* ni;
    loop.splitWithMask(n, 16, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* mi = loops[1];
    For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* ni = loops[3];
    For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* mi = loops[2];
    For* k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    loop.cacheAccesses(CT->buf(), "C_regs", loops[2]);
  }

  MemDependencyChecker analyzer_unlowered(
      loop.getInputBufs(), loop.getOutputBufs());

  MemDependencyChecker analyzer_lowered(
      loop.getInputBufs(), loop.getOutputBufs());

  // Test both unlowered and lowered form.
  {
    Stmt* stmt = IRSimplifier::simplify(loop.root_stmt());
    stmt->accept(&analyzer_unlowered);

    // Outputs depend on inputs.
    ASSERT_TRUE(analyzer_unlowered.dependsIndirectly(CT->buf(), AP.data()));
    ASSERT_TRUE(analyzer_unlowered.dependsIndirectly(CT->buf(), BP.data()));

    // The last write to gemm should cover the total bound of the output.
    std::shared_ptr<AccessInfo> outputAccess =
        analyzer_unlowered.output(CT->buf());
    // A single dependency.
    ASSERT_EQ(outputAccess->dependencies().size(), 1);

    // dependencies is a set with 1 element, so can just deref begin().
    std::shared_ptr<AccessInfo> gemmStore =
        outputAccess->dependencies().begin()->second;
    // Check its a store.
    ASSERT_EQ(gemmStore->type(), AccessType::Store);

    ASSERT_TRUE(indexBoundsEquals(outputAccess->bounds(), gemmStore->bounds()));

    // Likewise the first read from each input cover the entire range of the
    // input.
    auto aInput = analyzer_unlowered.input(AP.data());
    auto bInput = analyzer_unlowered.input(BP.data());

    // A single dependent each.
    ASSERT_EQ(aInput->dependents().size(), 1);
    ASSERT_EQ(bInput->dependents().size(), 1);

    // They're both loads.
    std::shared_ptr<AccessInfo> aLoad = aInput->dependents().begin()->second;
    std::shared_ptr<AccessInfo> bLoad = bInput->dependents().begin()->second;
    ASSERT_EQ(aLoad->type(), AccessType::Load);
    ASSERT_EQ(bLoad->type(), AccessType::Load);

    ASSERT_TRUE(indexBoundsEquals(aInput->bounds(), aLoad->bounds()));
    ASSERT_TRUE(indexBoundsEquals(bInput->bounds(), bLoad->bounds()));
  }

  loop.prepareForCodegen();

  // now check lowered dependency graph.
  {
    Stmt* stmt = IRSimplifier::simplify(loop.root_stmt());
    stmt->accept(&analyzer_lowered);

    // Lowering will change the dimensionality of all bounds due to index
    // flattening and will insert Allocates and Frees.

    auto history_before = analyzer_unlowered.getHistory();
    auto history_after = analyzer_lowered.getHistory();

    ASSERT_EQ(history_before.size() + 2, history_after.size());

    // Filter out the alloc/free;
    auto isAllocFree = [](const auto& info) {
      return info->type() == AccessType::Alloc ||
          info->type() == AccessType::Free;
    };
    history_after.erase(
        std::remove_if(history_after.begin(), history_after.end(), isAllocFree),
        history_after.end());

    ASSERT_EQ(history_before.size(), history_after.size());

    for (size_t i = 0; i < history_before.size(); ++i) {
      ASSERT_EQ(history_before[i]->type(), history_after[i]->type());
      ASSERT_EQ(history_before[i]->var(), history_after[i]->var());

      if (history_before[i]->dependencies().size() !=
          history_after[i]->dependencies().size()) {
        // Must depend on an Alloc.
        ASSERT_TRUE(std::any_of(
            history_after[i]->dependencies().begin(),
            history_after[i]->dependencies().end(),
            [](const auto& pair) {
              return pair.second->type() == AccessType::Alloc;
            }));

        ASSERT_EQ(
            history_before[i]->dependencies().size() + 1,
            history_after[i]->dependencies().size());
      }

      if (history_before[i]->dependents().size() !=
          history_after[i]->dependents().size()) {
        // Must depend on an Free.
        ASSERT_TRUE(std::any_of(
            history_after[i]->dependents().begin(),
            history_after[i]->dependents().end(),
            [](const auto& pair) {
              return pair.second->type() == AccessType::Free;
            }));

        ASSERT_EQ(
            history_before[i]->dependents().size() + 1,
            history_after[i]->dependents().size());
      }

      // Inputs and outputs are not flattened, only accesses.
      if (history_before[i]->type() == AccessType::Input ||
          history_before[i]->type() == AccessType::Output) {
        ASSERT_EQ(
            history_before[i]->bounds().size(),
            history_after[i]->bounds().size());
        ASSERT_TRUE(indexBoundsEquals(
            history_before[i]->bounds(), history_after[i]->bounds()));
      } else {
        ASSERT_EQ(history_after[i]->bounds().size(), 1);
        const Expr* flat_bounds = new IntImm(1);

        for (auto& b : history_before[i]->bounds()) {
          flat_bounds = new Mul(flat_bounds, new Add(b.end, new IntImm(1)));

          ASSERT_TRUE(exprEquals(b.start, history_after[i]->bounds()[0].start));
        }

        flat_bounds = IRSimplifier::simplify(flat_bounds);
        const Expr* after_bounds = IRSimplifier::simplify(
            new Add(history_after[i]->bounds()[0].end, new IntImm(1)));
        ASSERT_TRUE(exprEquals(flat_bounds, after_bounds));
      }
    }
  }
}

} // namespace jit
} // namespace torch

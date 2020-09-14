#include <test/cpp/tensorexpr/test_base.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/buffer.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/function.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

static void verifyConstBounds(
    const TensorAccessBoundsInfo& access_info,
    const std::vector<std::pair<int, int>>& ref) {
  size_t ndim = ref.size();
  ASSERT_EQ(access_info.start.size(), ndim);
  ASSERT_EQ(access_info.stop.size(), ndim);
  for (size_t i = 0; i < ndim; i++) {
    if (ref[i].first >= 0) { // Negative values are used to skip the check
      ASSERT_TRUE(access_info.start[i]->isConstant());
      int start_i = immediateAs<int>(access_info.start[i]);
      ASSERT_EQ(start_i, ref[i].first);
    }
    if (ref[i].second >= 0) {
      ASSERT_TRUE(access_info.stop[i]->isConstant());
      int stop_i = immediateAs<int>(access_info.stop[i]);
      ASSERT_EQ(stop_i, ref[i].second);
    }
  }
}

void testBoundsInference_1() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 99}}
  KernelScope kernel_scope;
  ExprHandle n(100);
  Buffer a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a(i); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.data())[0], {{0, 99}});

  ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 99}});
}

void testBoundsInference_2() {
  // Verify that bounds inference works for the following example:
  // for i in 0..n:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, n-1}, {a, kLoad, 0, n-1}}
  KernelScope kernel_scope;
  VarHandle n("n", kInt);
  Buffer a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a(i); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.data())[0], {{0, -1}});

  ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b->buf())[0], {{0, -1}});
}

void testBoundsInference_3() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i] * a[i+10]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 109}}
  KernelScope kernel_scope;
  ExprHandle n(100);
  Buffer a(BufHandle("a", {n + 10}, kFloat));
  Tensor* b = Compute(
      "b", {{n, "i"}}, [&](const VarHandle& i) { return a(i) * a(i + 10); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.data())[0], {{0, 109}});

  ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 99}});
}

void testBoundsInference_4() {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..200:
  //   for x in 0..320:
  //     c[y,x] = a[y,x] * b[y,x]
  KernelScope kernel_scope;
  ExprHandle W(320);
  ExprHandle H(200);
  Buffer a(BufHandle("a", {H, W}, kFloat));
  Tensor* b = Compute(
      "b", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return x * y;
      });
  Tensor* c = Compute(
      "c", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return a(y, x) * b->call(y, x);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  Stmt* body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 199}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop body's scope
    auto bounds_info = inferBounds(body);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {-1, -1}});
  }
}

void testBoundsInference_5() {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i]
  //
  // ==> split ==>
  //
  // for i_outer in 0..100/16:
  //   for i_inner in 0..16:
  //     b[i_outer * 16 + i_inner] = a[i_outer * 16 + i_inner]
  // for i_tail in 0..100%16:
  //   b[i_tail + (100/16)*16] = a[i_tail + (100/16)*16];
  KernelScope kernel_scope;
  ExprHandle n(100);
  Buffer a(BufHandle("a", {n}, kFloat));
  Tensor* b =
      Compute("b", {{n, "i"}}, [&](const VarHandle& i) { return a(i); });
  LoopNest l({b});

  For* outer;
  For* inner;
  For* tail;
  std::vector<For*> loops = l.getLoopStmtsFor(b);
  l.splitWithTail(loops[0], 16, &outer, &inner, &tail);

  {
    // Verify inferred bounds for the outer loop
    auto bounds_info = inferBounds(outer);
    ASSERT_EQ(bounds_info.size(), 2);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 95}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 95}});
  }
  {
    // Verify inferred bounds for the tail loop
    auto bounds_info = inferBounds(tail);
    ASSERT_EQ(bounds_info.size(), 2);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{96, 99}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{96, 99}});
  }
}

void testBoundsInference_6() {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..20:
  //   for x in 0..32:
  //     c[y,x] = a[y+100,x+100] * b[y*2,x*5]
  KernelScope kernel_scope;
  ExprHandle W(320);
  ExprHandle H(200);
  ExprHandle CW(32);
  ExprHandle CH(20);
  Buffer a(BufHandle("a", {H, W}, kFloat));
  Tensor* b = Compute(
      "b", {{H, "y"}, {W, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return x * y;
      });
  Tensor* c = Compute(
      "c", {{CH, "y"}, {CW, "x"}}, [&](const VarHandle& y, const VarHandle& x) {
        return a(y + 100, x + 100) * b->call(y * 2, x * 5);
      });
  LoopNest l({c});
  std::vector<For*> loops = l.getLoopStmtsFor(c);
  Stmt* body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{100, 119}, {100, 131}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 38}, {0, 155}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 19}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {100, 131}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {0, 155}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop body's scope
    auto bounds_info = inferBounds(body);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{-1, -1}, {-1, -1}});
  }
}

void testBoundsInferenceNonOverlapping() {
  KernelScope kernel_scope;
  ExprHandle H(3);
  Buffer a(BufHandle("a", {10}, kFloat));
  Tensor* b =
      Compute("b", {{H, "x"}}, [&](const VarHandle& x) { return a(x); });
  Tensor* c = Compute(
      "c", {{H, "x"}}, [&](const VarHandle& x) { return a(x + H + 1); });
  LoopNest l({b, c});
  std::vector<For*> loops = NodeFinder<For>::find(l.root_stmt());

  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0:2], writes to b[0:2]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 2}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 2}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0+4:2+4], writes to c[0:2]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{4, 6}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 2}});
  }
  {
    // Infer bounds on the high level program.
    auto bounds_info = inferBounds(l.root_stmt());
    ASSERT_EQ(bounds_info.size(), 3);

    // Should be union of above 2 bounds.
    ASSERT_EQ(bounds_info.at(a.data()).size(), 2);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 2}});
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[1], {{4, 6}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 2}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 2}});
  }
}

void testBoundsInferenceAdjacent() {
  KernelScope kernel_scope;
  ExprHandle H(6);
  Buffer a(BufHandle("a", {20}, kFloat));
  Tensor* b =
      Compute("b", {{H, "x"}}, [&](const VarHandle& x) { return a(x); });
  Tensor* c =
      Compute("c", {{H, "x"}}, [&](const VarHandle& x) { return a(x + H); });
  LoopNest l({b, c});
  std::vector<For*> loops = NodeFinder<For>::find(l.root_stmt());

  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0:5], writes to b[0:5]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 5}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 5}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0+6:5+6], writes to c[0:5]
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{6, 11}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 5}});
  }
  {
    // Infer bounds on the high level program.
    auto bounds_info = inferBounds(l.root_stmt());
    ASSERT_EQ(bounds_info.size(), 3);

    // Should be union of above 2 bounds, but this time the bounds of A can be
    // merged.
    ASSERT_EQ(bounds_info.at(a.data()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.data())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.data())[0], {{0, 11}});

    ASSERT_EQ(bounds_info.at(b->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b->buf())[0], {{0, 5}});

    ASSERT_EQ(bounds_info.at(c->buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c->buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c->buf())[0], {{0, 5}});
  }
}

void testMergeInferredBounds() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("a", {10}, kFloat));

  // There are seven cases to consider in mergeTensorAccesses(A, B)
  //   * A is lower than B and does not overlap.
  //   * A is higher than B and does not overlap.
  //   * A overlaps B on both ends.
  //   * B overlaps A on both ends.
  //   * A overlaps B on the lower end. (equiv to B overlaps A on upper end).
  //   * A overlaps B on the upper end. (likewise covers reverse)
  //   * A and B are the same range.

  BoundsInfo info;
  // Test no overlap, both ways.
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(3)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(7)}});
  info[a.data()].push_back({kLoad, {new IntImm(9)}, {new IntImm(9)}});
  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[a.data()].size(), 3);

  ASSERT_EQ(res.at(a.data())[0].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[1].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[2].kind, kLoad);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{5, 7}});
  verifyConstBounds(res.at(a.data())[2], {{9, 9}});

  // Test full overlap, A over B.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(7)}});
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});

  // B over A.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(7)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});

  // Test partial overlap on the low end, A over B.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(7)}});
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{3, 7}});

  // Test partial overlap on the high end.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(2)}, {new IntImm(5)}});
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{2, 6}});

  // Test equality is deduped.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{4, 6}});
}

void testMergeInferredLoadStoreDiff() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("a", {10}, kFloat));

  // Loads and Stores do not merge:
  BoundsInfo info;
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(7)}});
  info[a.data()].push_back({kStore, {new IntImm(3)}, {new IntImm(9)}});

  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[a.data()].size(), 2);
  ASSERT_EQ(res.at(a.data())[0].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[1].kind, kStore);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});
  verifyConstBounds(res.at(a.data())[1], {{3, 9}});

  // Do merge around the other kind of access:
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(3)}});
  info[a.data()].push_back({kStore, {new IntImm(3)}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(3)}, {new IntImm(5)}});
  info[a.data()].push_back({kStore, {new IntImm(4)}, {new IntImm(8)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(7)}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 7}});
  verifyConstBounds(res.at(a.data())[1], {{3, 8}});
}

void testMergeInferred2DBounds() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("a", {10, 10}, kFloat));

  // Non overlapping in both dimensions:
  BoundsInfo info;
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(3), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(5), new IntImm(5)}, {new IntImm(9), new IntImm(9)}});

  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res.size(), 1);
  ASSERT_EQ(res[a.data()].size(), 2);
  ASSERT_EQ(res.at(a.data())[0].kind, kLoad);
  ASSERT_EQ(res.at(a.data())[1].kind, kLoad);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}, {1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{5, 9}, {5, 9}});

  // Overlapping in a single dimension should mean we cannot merge.
  // First dimension:
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(3), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(2), new IntImm(5)}, {new IntImm(9), new IntImm(9)}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}, {1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{2, 9}, {5, 9}});

  // Second dimension:
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(3), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(5), new IntImm(2)}, {new IntImm(9), new IntImm(9)}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 3}, {1, 3}});
  verifyConstBounds(res.at(a.data())[1], {{5, 9}, {2, 9}});

  // Overlapping in both dimensions:
  // {1-6, 1-3) | {4-9, 2,7} => {1,9, 1,7}
  // TODO: this will overestimate and we should fix it.
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new IntImm(1), new IntImm(1)}, {new IntImm(6), new IntImm(3)}});
  info[a.data()].push_back(
      {kLoad, {new IntImm(4), new IntImm(2)}, {new IntImm(9), new IntImm(7)}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 9}, {1, 7}});
}

void testMergeAdjacentBounds() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("a", {10}, kFloat));

  // Adjacent but not overlapping bounds can be merged.
  // e.g. {1-4} | {5-9} => {1-9}
  BoundsInfo info;
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(9)}});
  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 9}});

  // And on the other side:
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {new IntImm(9)}});
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(4)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  verifyConstBounds(res.at(a.data())[0], {{1, 9}});

  // One space gap is enough to prevent merging:
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(1)}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {new IntImm(9)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
  verifyConstBounds(res.at(a.data())[0], {{1, 4}});
  verifyConstBounds(res.at(a.data())[1], {{6, 9}});
}

std::pair<std::string, std::string> boundAsStringPair(
    TensorAccessBoundsInfo& info,
    size_t idx = 0) {
  std::ostringstream start, stop;
  start << *info.start[idx];
  stop << *info.stop[idx];
  return {start.str(), stop.str()};
}

void testMergeSymbolicBounds() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("a", {10}, kFloat));
  VarHandle W("W", kInt);
  VarHandle X("X", kInt);
  VarHandle Y("Y", kInt);
  VarHandle Z("Z", kInt);

  // Can do nothing with fully symbolic bounds:
  BoundsInfo info;
  info[a.data()].push_back({kLoad, {W.node()}, {Z.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  BoundsInfo res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // Can merge if the difference between bounds is constant and enclosing.
  // {X-Y} | {X-5 - Y+10} => {X-5 - Y+10}
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back({kLoad,
                            {new Sub(X.node(), new IntImm(5))},
                            {new Add(Y.node(), new IntImm(10))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);

  // Cannot merge otherwise.
  // {X-Y} | {X+5 - Y+10} => could be 2 groups if Y < X+5.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back({kLoad,
                            {new Add(X.node(), new IntImm(5))},
                            {new Add(Y.node(), new IntImm(10))}});

  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // Can't merge if there's a gap of at least one element:
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(4)}});
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // Can't even though the high of the first bound is above the low of the
  // second, X can == 6 and Y can == 4 so this can't merge in all cases.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(4)}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // If either side is equal, they must be overlapping.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Z.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  auto pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Max(Y, Z, 1)");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back({kLoad, {Z.node()}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "Min(Z, X, 1)");
  ASSERT_EQ(pair.second, "Y");

  // If either side is only one apart, they must be adjacent.
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new Add(X.node(), new IntImm(1))}, {Z.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Max(Y, Z, 1)");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back(
      {kLoad, {Z.node()}, {new Sub(Y.node(), new IntImm(1))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "Min(Z, X, 1)");
  ASSERT_EQ(pair.second, "Y");

  // If either side is 2 apart, they may not be overlapping.
  // in this case if Y == X+1 they don't overlap.
  info.clear();
  info[a.data()].push_back(
      {kLoad, {new Add(X.node(), new IntImm(2))}, {Z.node()}});
  info[a.data()].push_back(
      {kLoad, {X.node()}, {new Sub(Y.node(), new IntImm(1))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);

  // In this case they may not overlap if X == Y.
  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {Y.node()}});
  info[a.data()].push_back(
      {kLoad, {Z.node()}, {new Sub(Y.node(), new IntImm(2))}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 2);
}

void testMergeSymbolicAdjacent() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("a", {10}, kFloat));
  VarHandle X("X", kInt);
  VarHandle Y("Y", kInt);

  BoundsInfo info;
  // Can merge if a range is adjacent:
  // {X-5} | {6-Y} => {X-Y}
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(5)}});
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {Y.node()}});
  BoundsInfo res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  auto pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {Y.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(5)}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {Y.node()}});
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {Y.node()}});
  res = mergeTensorAccesses(info);

  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "X");
  ASSERT_EQ(pair.second, "Y");

  // If either the lower or upper bound is adjacent the range then they must
  // overlap, even if we don't know the extent.
  info.clear();
  info[a.data()].push_back({kLoad, {new IntImm(6)}, {X.node()}});
  info[a.data()].push_back({kLoad, {new IntImm(5)}, {Y.node()}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "5");
  ASSERT_EQ(pair.second, "Max(Y, X, 1)");

  info.clear();
  info[a.data()].push_back({kLoad, {X.node()}, {new IntImm(6)}});
  info[a.data()].push_back({kLoad, {Y.node()}, {new IntImm(5)}});
  res = mergeTensorAccesses(info);
  ASSERT_EQ(res[a.data()].size(), 1);
  pair = boundAsStringPair(res[a.data()][0]);
  ASSERT_EQ(pair.first, "Min(Y, X, 1)");
  ASSERT_EQ(pair.second, "6");
}

} // namespace jit
} // namespace torch

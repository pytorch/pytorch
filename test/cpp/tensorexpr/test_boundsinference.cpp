#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/bounds_inference.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
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
  for (const auto i : c10::irange(ndim)) {
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

TEST(BoundsInference, _1) {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 99}}
  ExprHandle n(100);
  BufHandle a("a", {n}, kFloat);
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.node())[0], {{0, 99}});

  ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 99}});
}

TEST(BoundsInference, _2) {
  // Verify that bounds inference works for the following example:
  // for i in 0..n:
  //   b[i] = a[i]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, n-1}, {a, kLoad, 0, n-1}}
  VarHandle n("n", kInt);
  BufHandle a("a", {n}, kFloat);
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.node())[0], {{0, -1}});

  ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b.buf())[0], {{0, -1}});
}

TEST(BoundsInference, _3) {
  // Verify that bounds inference works for the following example:
  // for i in 0..100:
  //   b[i] = a[i] * a[i+10]
  // For this loop bounds inference should yield the following:
  // {{b, kStore, 0, 99}, {a, kLoad, 0, 109}}
  ExprHandle n(100);
  BufHandle a("a", {n + 10}, kFloat);
  Tensor b = Compute(
      "b", {n}, [&](const VarHandle& i) { return a.load(i) * a.load(i + 10); });
  LoopNest l({b});
  auto bounds_info = inferBounds(l.root_stmt());

  // We should have two entries: one for 'b' and one for 'a'.
  ASSERT_EQ(bounds_info.size(), 2);
  ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
  ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
  verifyConstBounds(bounds_info.at(a.node())[0], {{0, 109}});

  ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
  ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
  verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 99}});
}

TEST(BoundsInference, _4) {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..200:
  //   for x in 0..320:
  //     c[y,x] = a[y,x] * b[y,x]
  ExprHandle W(320);
  ExprHandle H(200);
  BufHandle a("a", {H, W}, kFloat);
  Tensor b = Compute("b", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
    return x * y;
  });
  Tensor c = Compute("c", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
    return a.load(y, x) * b.load(y, x);
  });
  LoopNest l({c});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  StmtPtr body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 199}, {0, 319}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{0, 199}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {0, 319}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {0, 319}});
  }
  {
    // Infer bounds on the inner loop body's scope
    auto bounds_info = inferBounds(body);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {-1, -1}});
  }
}

TEST(BoundsInference, _5) {
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
  ExprHandle n(100);
  BufHandle a("a", {n}, kFloat);
  Tensor b = Compute("b", {n}, [&](const VarHandle& i) { return a.load(i); });
  LoopNest l({b});

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr tail;
  std::vector<ForPtr> loops = l.getLoopStmtsFor(b);
  LoopNest::splitWithTail(loops[0], 16, &inner, &tail);
  ForPtr outer = loops[0];

  {
    // Verify inferred bounds for the outer loop
    auto bounds_info = inferBounds(outer);
    ASSERT_EQ(bounds_info.size(), 2);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{0, 95}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 95}});
  }
  {
    // Verify inferred bounds for the tail loop
    auto bounds_info = inferBounds(tail);
    ASSERT_EQ(bounds_info.size(), 2);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{96, 99}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{96, 99}});
  }
}

TEST(BoundsInference, _6) {
  // Verify that bounds inference works for the following example:
  //
  // for y in 0..200:
  //   for x in 0..320:
  //     b[y,x] = x*y
  // for y in 0..20:
  //   for x in 0..32:
  //     c[y,x] = a[y+100,x+100] * b[y*2,x*5]
  ExprHandle W(320);
  ExprHandle H(200);
  ExprHandle CW(32);
  ExprHandle CH(20);
  BufHandle a("a", {H, W}, kFloat);
  Tensor b = Compute("b", {H, W}, [&](const VarHandle& y, const VarHandle& x) {
    return x * y;
  });
  Tensor c =
      Compute("c", {CH, CW}, [&](const VarHandle& y, const VarHandle& x) {
        return a.load(y + 100, x + 100) * b.load(y * 2, x * 5);
      });
  LoopNest l({c});
  std::vector<ForPtr> loops = l.getLoopStmtsFor(c);
  StmtPtr body = l.getLoopBodyFor(c);
  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{100, 119}, {100, 131}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 38}, {0, 155}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{0, 19}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {100, 131}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {0, 155}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {0, 31}});
  }
  {
    // Infer bounds on the inner loop body's scope
    auto bounds_info = inferBounds(body);
    ASSERT_EQ(bounds_info.size(), 3);

    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{-1, -1}, {-1, -1}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{-1, -1}, {-1, -1}});
  }
}

TEST(BoundsInference, Adjacent) {
  ExprHandle H(6);
  BufHandle a("a", {20}, kFloat);
  Tensor b = Compute("b", {H}, [&](const VarHandle& x) { return a.load(x); });
  Tensor c =
      Compute("c", {H}, [&](const VarHandle& x) { return a.load(x + H); });
  LoopNest l({b, c});
  std::vector<ForPtr> loops = NodeFinder<For>::find(l.root_stmt());

  {
    // Infer bounds on the top-level loop scope
    auto bounds_info = inferBounds(loops[0]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0:5], writes to b[0:5]
    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{0, 5}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 5}});
  }
  {
    // Infer bounds on the inner loop scope
    auto bounds_info = inferBounds(loops[1]);
    ASSERT_EQ(bounds_info.size(), 2);

    // reads from a[0+6:5+6], writes to c[0:5]
    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{6, 11}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{0, 5}});
  }
  {
    // Infer bounds on the high level program.
    auto bounds_info = inferBounds(l.root_stmt());
    ASSERT_EQ(bounds_info.size(), 3);

    // Should be union of above 2 bounds, but this time the bounds of A can be
    // merged.
    ASSERT_EQ(bounds_info.at(a.node()).size(), 1);
    ASSERT_EQ(bounds_info.at(a.node())[0].kind, kLoad);
    verifyConstBounds(bounds_info.at(a.node())[0], {{0, 11}});

    ASSERT_EQ(bounds_info.at(b.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(b.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(b.buf())[0], {{0, 5}});

    ASSERT_EQ(bounds_info.at(c.buf()).size(), 1);
    ASSERT_EQ(bounds_info.at(c.buf())[0].kind, kStore);
    verifyConstBounds(bounds_info.at(c.buf())[0], {{0, 5}});
  }
}

TEST(BoundsInference, MultipleTopLoopLoad) {
  BufHandle a("a", {100}, kFloat);
  Tensor b = Compute("b", {64}, [&](const VarHandle& x) { return a.load(x); });
  Tensor c =
      Compute("c", {32}, [&](const VarHandle& x) { return a.load(x + 10); });
  Tensor d =
      Compute("d", {96}, [&](const VarHandle& x) { return a.load(x + 2); });
  LoopNest l({b, c, d});

  auto bounds_info = inferBounds(l.root_stmt());

  ASSERT_EQ(bounds_info.size(), 4);

  // a only read.
  {
    auto bounds = bounds_info[a.node()];
    ASSERT_EQ(bounds.size(), 1);
    // One dimension.
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kLoad);
    // Bounds:
    // start: Min of the 3 load bounds = Min of loop starts + offset = 0+0 (b).
    // stop: Max of the 3 load bounds = Max of loop stops + offset - 1 =
    //       96 + 2 - 1 (d).
    verifyConstBounds(bound, {{0, 97}});
  }

  // b, c, d only written.
  {
    auto bounds = bounds_info[b.buf()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // Just the loop extents for b.
    verifyConstBounds(bound, {{0, 63}});
  }
  {
    auto bounds = bounds_info[c.buf()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // Just the loop extents for c.
    verifyConstBounds(bound, {{0, 31}});
  }
  {
    auto bounds = bounds_info[d.buf()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // Just the loop extents for d.
    verifyConstBounds(bound, {{0, 95}});
  }
}

TEST(BoundsInference, MultipleTopLoopStore) {
  BufHandle a("a", {100}, kFloat);
  BufHandle b("b", {100}, kFloat);
  BufHandle c("c", {100}, kFloat);
  BufHandle d("d", {100}, kFloat);
  VarHandle x("x", kInt);

  // Same as above but the offsets are on the Store now.
  // Can't do this through ComputeAPI without transforms we don't have yet.
  StmtPtr stmt = Block::make(
      {For::make(x, 0, 64, Store::make(b, {x}, Load::make(a, {x}))),
       For::make(x, 0, 32, Store::make(c, {x + 10}, Load::make(a, {x}))),
       For::make(x, 0, 96, Store::make(d, {x + 2}, Load::make(a, {x})))});

  auto bounds_info = inferBounds(stmt);

  ASSERT_EQ(bounds_info.size(), 4);

  // a only read.
  {
    auto bounds = bounds_info[a.node()];
    ASSERT_EQ(bounds.size(), 1);
    // One dimension.
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kLoad);
    // Bounds: there are no offsets, so this is just the max loop bounds.
    verifyConstBounds(bound, {{0, 95}});
  }

  // b, c, d only written.
  {
    auto bounds = bounds_info[b.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // This should be equivalent to {offset, extent + offset} for the b loop.
    // b loop has no offset, so just the loop extents.
    verifyConstBounds(bound, {{0, 63}});
  }
  {
    auto bounds = bounds_info[c.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // This should be equivalent to {offset, extent + offset} for the c loop.
    // Offset is 10, extent is 32-1.
    verifyConstBounds(bound, {{10, 41}});
  }
  {
    auto bounds = bounds_info[d.node()];
    ASSERT_EQ(bounds.size(), 1);
    auto bound = bounds[0];
    ASSERT_EQ(bound.kind, TensorAccessKind::kStore);
    // This should be equivalent to {offset, extent + offset} for the d loop.
    // Offset is 2, extent is 96-1.
    verifyConstBounds(bound, {{2, 97}});
  }
}

TEST(BoundsInference, CacheReads) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B =
      Compute("B", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 30, j + 3);
      });
  Tensor C =
      Compute("C", {20, 10}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i + 10, j + 20) + A.load(i + 30, j + 40);
      });

  LoopNest l({B, C});
  auto bounds_info_before = inferBounds(l.root_stmt());

  StmtPtr j_loop = l.getLoopStmtsFor(B)[1];
  LoopNest::cacheAccesses(A.buf(), "A_local", j_loop);

  auto bounds_info_after = inferBounds(l.root_stmt());

  // CacheAccesses should not change existing bounds, but add a new one for the
  // cache.
  for (auto& pair : bounds_info_after) {
    auto beforeIt = bounds_info_before.find(pair.first);
    if (beforeIt != bounds_info_before.end()) {
      // Same number of TensorAccessBoundInfos.
      ASSERT_EQ(pair.second.size(), beforeIt->second.size());

      for (const auto i : c10::irange(pair.second.size())) {
        TensorAccessBoundsInfo& after = pair.second[i];
        TensorAccessBoundsInfo& before = beforeIt->second[i];
        // Same number of dimensions.
        ASSERT_EQ(before.start.size(), after.start.size());

        // Bounds are equal.
        for (const auto j : c10::irange(before.start.size())) {
          ASSERT_TRUE(exprEquals(before.start[j], after.start[j]));
          ASSERT_TRUE(exprEquals(before.stop[j], after.stop[j]));
        }
      }
    } else {
      // This should be the cache.
      ASSERT_EQ(pair.first->name_hint(), "A_local");
      // Should have both a load and a store.
      ASSERT_EQ(pair.second.size(), 2);
      TensorAccessBoundsInfo& first = pair.second[0];
      TensorAccessBoundsInfo& second = pair.second[1];

      ASSERT_NE(first.kind, second.kind);
      // 2 dimensions.
      ASSERT_EQ(first.start.size(), second.start.size());
      ASSERT_EQ(first.start.size(), 2);

      // bounds for load and store are equal.
      for (const auto j : c10::irange(first.start.size())) {
        ASSERT_TRUE(exprEquals(first.start[j], second.start[j]));
        ASSERT_TRUE(exprEquals(first.stop[j], second.stop[j]));
      }
    }
  }
}

TEST(BoundsInference, Flattened) {
  Tensor b = Compute(
      "b",
      {3, 4, 5},
      [&](const VarHandle& z, const VarHandle& y, const VarHandle& x) {
        return x * y + z;
      });

  LoopNest l({b});
  // Flatten indices.
  l.prepareForCodegen();
  auto bounds_info = inferBounds(l.root_stmt());

  // There's only one buffer.
  ASSERT_EQ(bounds_info.size(), 1);
  auto& TABI = bounds_info[b.buf()][0];
  ASSERT_EQ(TABI.kind, TensorAccessKind::kStore);
  // Flattened bounds should have a single dimension.
  ASSERT_EQ(TABI.start.size(), 1);
  ASSERT_EQ(TABI.stop.size(), 1);

  // Bounds should be 0 -> (3*4*5)-1
  ASSERT_TRUE(exprEquals(TABI.start[0], alloc<IntImm>(0)));
  ASSERT_TRUE(exprEquals(TABI.stop[0], alloc<IntImm>(3 * 4 * 5 - 1)));
}

TEST(BoundsInference, GetPotentialHazards) {
  BufHandle a("A", {5}, kInt);
  BufHandle b("B", {5}, kInt);
  BufHandle c("C", {5}, kInt);
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  using namespace analysis;

  {
    /*
     * A[0] = B[0];
     * B[0] = 3;      WAR on B
     * A[0] = B[0];   WAW on A, RAW on B
     * C[0] = 5;
     */

    StorePtr store1 = Store::make(a, {0}, Load::make(b, {0}));
    StorePtr store2 = Store::make(b, {0}, 3);
    StorePtr store3 = Store::make(a, {0}, Load::make(b, {0}));
    StorePtr store4 = Store::make(c, {0}, 5);
    StmtPtr stmt = Block::make({store1, store2, store3, store4});

    MemDependencyChecker analyzer;
    stmt->accept(&analyzer);

    ASSERT_EQ(
        HazardKind::WriteAfterRead,
        getPotentialHazards(analyzer, store1, store2));

    ASSERT_EQ(
        HazardKind::ReadAfterWrite,
        getPotentialHazards(analyzer, store2, store3));

    ASSERT_EQ(
        HazardKind::WriteAfterWrite,
        getPotentialHazards(analyzer, store1, store3));

    // Fourth store has no dependencies
    ASSERT_EQ(
        HazardKind::NoDependency,
        getPotentialHazards(analyzer, store1, store4));
    ASSERT_EQ(
        HazardKind::NoDependency,
        getPotentialHazards(analyzer, store2, store4));
    ASSERT_EQ(
        HazardKind::NoDependency,
        getPotentialHazards(analyzer, store3, store4));
  }
}

TEST(BoundsInference, GetPotentialHazardsLoopNoHazard) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B = Compute("B", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return (i + 1) * (j + 1);
  });

  LoopNest l({A, B});

  using namespace analysis;

  MemDependencyChecker analyzer;
  l.root_stmt()->accept(&analyzer);

  ForPtr loopRootA = l.getLoopStmtsFor(A)[0];
  ForPtr loopRootB = l.getLoopStmtsFor(B)[0];

  // No dependencies between loops.
  ASSERT_EQ(
      HazardKind::NoDependency,
      getPotentialHazards(analyzer, loopRootA, loopRootB));
}

TEST(BoundsInference, GetPotentialHazardsLoopCall) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });
  Tensor B =
      Compute("B", {64, 64}, [&](const VarHandle& i, const VarHandle& j) {
        return A.load(i, j) + 5;
      });

  LoopNest l({A, B});

  using namespace analysis;

  MemDependencyChecker analyzer;
  l.root_stmt()->accept(&analyzer);

  ForPtr loopRootA = l.getLoopStmtsFor(A)[0];
  ForPtr loopRootB = l.getLoopStmtsFor(B)[0];

  ASSERT_EQ(
      HazardKind::ReadAfterWrite,
      getPotentialHazards(analyzer, loopRootA, loopRootB));
}

TEST(BoundsInference, GetPotentialHazardsLoopSplit) {
  Tensor A = Compute("A", {64, 64}, [](const VarHandle& i, const VarHandle& j) {
    return i * j;
  });

  LoopNest l({A});
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ForPtr inner, tail;

  // Splitting with tail by something offset creates a tail which also writes to
  // A.
  ForPtr outer = l.getLoopStmtsFor(A)[0];
  // `outer` loop get transformed to the outer loop after splitting.
  LoopNest::splitWithTail(outer, 5, &inner, &tail);

  using namespace analysis;

  MemDependencyChecker analyzer;
  l.root_stmt()->accept(&analyzer);

  ASSERT_EQ(
      HazardKind::WriteAfterWrite, getPotentialHazards(analyzer, outer, tail));
}

TEST(BoundsInference, HasConflictingOverlapSameBufferWithPartialOverlap) {
  // Input IR:
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     A[k-1] = 20 * k;
  //   }
  BufHandle a_buf("A", {200}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k - 1}, Mul::make(20, k)));
  auto par = Block::make({forJ, forK});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapSameBufferWithFullOverlap) {
  // Input IR:
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     A[k] = 20 * k;
  //   }
  BufHandle a_buf("A", {200}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(k, 10, 100, Store::make(a_buf, {k}, Mul::make(20, k)));
  auto par = Block::make({forJ, forK});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapSameBufferWithFullOverlapRAW) {
  // Input IR:
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     B[k] = A[k];
  //   }
  BufHandle a_buf("A", {200}, kInt);
  BufHandle b_buf("B", {200}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK =
      For::make(k, 10, 100, Store::make(b_buf, {k}, Load::make(a_buf, {k})));
  auto par = Block::make({forJ, forK});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapSameBufferNotOverlapping) {
  // Input IR:
  //   for (const auto j : c10::irange(10, 100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(10, 100)) {
  //     A[k+100] = 20 * k;
  //   }
  BufHandle a_buf("A", {200}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 10, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK =
      For::make(k, 10, 100, Store::make(a_buf, {k + 100}, Mul::make(20, k)));
  auto par = Block::make({forJ, forK});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlap2DBufferWithOverlap) {
  // Input IR:
  //   for (const auto i : c10::irange(20)) {
  //     for (const auto j : c10::irange(100)) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (const auto m : c10::irange(20)) {
  //     for (const auto n : c10::irange(50)) {
  //       A[m+1,n] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 50}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  auto forJ = For::make(j, 0, 100, storeA1);
  auto forI = For::make(i, 0, 20, forJ);
  auto storeA2 =
      Store::make(a_buf, {m + 1, n}, Add::make(m, Mul::make(n, 100)));
  auto forN = For::make(n, 0, 50, storeA2);
  auto forM = For::make(m, 0, 20, forN);
  auto par = Block::make({forI, forM});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forI, forM));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forM, forI));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forN));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forN, forJ));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, storeA1, storeA2));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, storeA2, storeA1));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, storeA2));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, storeA1, forM));
}

TEST(BoundsInference, HasConflictingOverlap2DBufferWithNoOverlap) {
  // Input IR:
  //   for (const auto i : c10::irange(20)) {
  //     for (const auto j : c10::irange(100)) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (const auto m : c10::irange(20)) {
  //     for (const auto n : c10::irange(50)) {
  //       A[m+20,n+100] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 50}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  auto forJ = For::make(j, 0, 100, storeA1);
  auto forI = For::make(i, 0, 20, forJ);
  auto storeA2 =
      Store::make(a_buf, {m + 20, n + 100}, Add::make(m, Mul::make(n, 100)));
  auto forN = For::make(n, 0, 50, storeA2);
  auto forM = For::make(m, 0, 20, forN);
  auto par = Block::make({forI, forM});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forI, forM));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forM, forI));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forN));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forN, forJ));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA2, storeA1));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, forM));
}

TEST(BoundsInference, HasConflictingOverlapDifferentBuffers) {
  // Input IR:
  //   for (const auto i : c10::irange(20)) {
  //     for (const auto j : c10::irange(100)) {
  //       A[i,j] = i * j * 500;
  //     }
  //   }
  //   for (const auto m : c10::irange(20)) {
  //     for (const auto n : c10::irange(50)) {
  //       B[m,n] = m + n * 100;
  //     }
  //   }
  BufHandle a_buf("A", {20, 100}, kInt);
  BufHandle b_buf("B", {20, 50}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  auto storeA1 = Store::make(a_buf, {i, j}, Mul::make(Mul::make(i, j), 500));
  auto forJ = For::make(j, 0, 100, storeA1);
  auto forI = For::make(i, 0, 20, forJ);
  auto storeA2 = Store::make(b_buf, {m, n}, Add::make(m, Mul::make(n, 100)));
  auto forN = For::make(n, 0, 50, storeA2);
  auto forM = For::make(m, 0, 20, forN);
  auto par = Block::make({forI, forM});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forI, forM));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forM, forI));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forN));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forN, forJ));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA2, storeA1));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, storeA2));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, storeA1, forM));
}

TEST(BoundsInference, HasConflictingOverlapDueToRAWDependence) {
  // Input IR:
  //   for (const auto j : c10::irange(100)) {
  //     A[j] = 10 * j;
  //   }
  //   for (const auto k : c10::irange(100)) {
  //     B[k] = 20 * A[99-k];
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  auto par = Block::make({forJ, forK});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapDueToWARDependence) {
  // Input IR:
  //   for (const auto k : c10::irange(100)) {
  //     B[k] = 20 * A[99-k];
  //   }
  //   for (const auto j : c10::irange(100)) {
  //     A[j] = 10 * j;
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forK = For::make(
      k,
      0,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  auto forJ = For::make(j, 0, 100, Store::make(a_buf, {j}, Mul::make(10, j)));
  auto par = Block::make({forK, forJ});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_TRUE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, HasConflictingOverlapWithLoads) {
  // Input IR:
  //   for (const auto k : c10::irange(10, 100)) {
  //     B[k] = 20 * A[99-k];
  //   }
  //   for (const auto j : c10::irange(10, 100)) {
  //     C[j] = 10 * A[j];
  //   }
  BufHandle a_buf("A", {100}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  BufHandle c_buf("C", {100}, kInt);
  VarHandle j("j", kInt);
  VarHandle k("k", kInt);
  auto forK = For::make(
      k,
      10,
      100,
      Store::make(
          b_buf, {k}, Mul::make(20, Load::make(a_buf, {ExprHandle(99) - k}))));
  auto forJ = For::make(
      j,
      10,
      100,
      Store::make(c_buf, {j}, Mul::make(10, Load::make(a_buf, {j}))));
  auto par = Block::make({forK, forJ});

  tensorexpr::analysis::MemDependencyChecker analyzer;
  par->accept(&analyzer);
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forJ, forK));
  ASSERT_FALSE(hasConflictingOverlap(analyzer, forK, forJ));
}

TEST(BoundsInference, IsOverlapping) {
  // Input IR:
  //   for (const auto i : c10::irange(100)) {
  //     A[i] = i * 10;               // storeA1
  //     B[i] = A[99-i] * 20;         // loadA1
  //     C[i] = A[i + 100] * 10;      // loadA2
  //     A[i + 50] = i * 50;          // storeA2
  //     A[i + 150] = i * 150;        // storeA3
  //   }
  BufHandle a_buf("A", {300}, kInt);
  BufHandle b_buf("B", {100}, kInt);
  BufHandle c_buf("C", {100}, kInt);
  VarHandle i("i", kInt);
  auto storeA1 = Store::make(a_buf, {i}, i * 10);
  auto loadA1 = Load::make(a_buf, {ExprHandle(99) - i});
  auto storeB = Store::make(b_buf, {i}, Mul::make(loadA1, 20));
  auto loadA2 = Load::make(a_buf, {i + 100});
  auto storeC = Store::make(c_buf, {i}, Mul::make(loadA2, 10));
  auto storeA2 = Store::make(a_buf, {i + 50}, i * 50);
  auto storeA3 = Store::make(a_buf, {i + 150}, i * 150);
  auto forI = For::make(
      i, 0, 100, Block::make({storeA1, storeB, storeC, storeA2, storeA3}));
  tensorexpr::analysis::MemDependencyChecker analyzer;
  forI->accept(&analyzer);
  ASSERT_TRUE(isOverlapping(analyzer, storeA1, to<Load>(loadA1.node())));
  ASSERT_FALSE(isOverlapping(analyzer, storeA1, to<Load>(loadA2.node())));
  ASSERT_TRUE(isOverlapping(analyzer, storeA1, storeA2));
  ASSERT_FALSE(isOverlapping(analyzer, storeA1, storeA3));
}

} // namespace jit
} // namespace torch

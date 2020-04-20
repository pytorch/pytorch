#include "test/cpp/tensorexpr/test_base.h"

#include <torch/csrc/jit/tensorexpr/cost_estimator.h>
#include "test/cpp/tensorexpr/test_utils.h"

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

void testCostEstimatorSimple() {
  KernelScope kernel_scope;
  Expr* f =
      new Add(getImmediateByType(kFloat, 2.f), getImmediateByType(kFloat, 3.f));

  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(f);

  // There are 3 Exprs, but only one has cost.
  ASSERT_TRUE(cost->isConstant());
  ASSERT_EQ(immediateAs<int>(cost), 1);
}

void testCostEstimatorCompound() {
  KernelScope kernel_scope;
  ExprHandle a(59);
  ExprHandle b(22);
  ExprHandle c(101);
  ExprHandle f = (a ^ b) & c;

  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(f.node());

  // Two binary ops.
  ASSERT_TRUE(cost->isConstant());
  ASSERT_EQ(immediateAs<int>(cost), 2);
}

void testCostEstimatorStaticFor() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {4}, kInt));
  Buffer c(BufHandle("C", {4}, kInt));
  auto mask = IntImm::make(1);
  VarHandle i("i", kInt);
  auto body =
      For::make(i, 0, 10, Store::make(c, {i}, Load::make(a, {i}, mask), mask));

  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(body);

  // 10 loop * store + load + 10 increments + 11 compare with stop.
  ASSERT_TRUE(cost->isConstant());
  ASSERT_EQ(immediateAs<int>(cost), 271);
}

void testCostEstimatorVariableFor() {
  KernelScope kernel_scope;
  Buffer a(BufHandle("A", {4}, kInt));
  Buffer c(BufHandle("C", {4}, kInt));
  auto mask = IntImm::make(1);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto body =
      For::make(i, 0, j, Store::make(c, {i}, Load::make(a, {i}, mask), mask));

  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(body);

  // j loops * (store + load = 25) + j increments + (j+1) compares:
  //  25 * j + j + j + 1 => 27 * j + 1.
  ASSERT_FALSE(cost->isConstant());

  ExprHandle expected = ExprHandle(27) * j + ExprHandle(1);

  // Compare via hash.
  HashProvider hasher;
  ASSERT_EQ(hasher.hash(cost), hasher.hash(expected.node()));
}

void testCostEstimatorBlock() {
  KernelScope kernel_scope;
  Buffer c(BufHandle("C", {4}, kInt));
  auto mask = IntImm::make(1);

  auto store1 = Store::make(c, {0}, IntImm::make(0), mask);
  auto store2 = Store::make(c, {2}, IntImm::make(2), mask);
  auto store3 = Store::make(c, {3}, IntImm::make(3), mask);
  auto store4 = Store::make(c, {1}, IntImm::make(1), mask);

  auto body = Block::make({store1, store2, store3, store4});
  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(body);

  // 4 store (15 cost) statements = 60.
  ASSERT_TRUE(cost->isConstant());
  ASSERT_EQ(immediateAs<int>(cost), 60);
}

void testCostEstimatorLoadStoreIndices() {
  KernelScope kernel_scope;
  Buffer c(BufHandle("C", {5, 5, 5}, kInt));
  auto mask = IntImm::make(1);

  auto idx1 = ExprHandle(1) + ExprHandle(2);
  auto idx2 = ExprHandle(2) * ExprHandle(2);
  auto idx3 = ExprHandle(10) / ExprHandle(2);

  auto store1 = Store::make(c, {idx1, idx2, idx3}, IntImm::make(0), mask);

  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(store1);

  // store (15 costs) + indexes (1, 1 and 1) => 18.
  ASSERT_TRUE(cost->isConstant());
  ASSERT_EQ(immediateAs<int>(cost), 18);

  VarHandle i("i", kInt);
  auto body = For::make(i, 0, 5, store1);
  const Expr* cost2 = estimator.estimateCost(body);

  // Cost is 5 * store (18) + 5 + (5+1) => 101.
  ASSERT_TRUE(cost2->isConstant());
  ASSERT_EQ(immediateAs<int>(cost2), 101);
}

void testCostEstimatorCond() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle i("i", kInt);

  Buffer c(BufHandle("C", {4}, kInt));
  auto body = Cond::make(
      x > y,
      For::make(
          i, 0, 10, Store::make(c, {0}, IntImm::make(1), IntImm::make(1))),
      For::make(
          i, 0, 3, Store::make(c, {1}, IntImm::make(1), IntImm::make(1))));

  // true branch is cost 10 * 15 + 20, false branch is cost 3 * 15 + 7. We don't
  // know which branch to take so assume worse case. total cost then is
  // max(10*15, 3
  // * 15) + 1 (for the compare).
  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(body);

  ASSERT_TRUE(cost->isConstant());
  ASSERT_EQ(immediateAs<int>(cost), 172);
}

void testCostEstimatorAllocFree() {
  KernelScope kernel_scope;
  Buffer c(BufHandle("C", {5, 5, 5}, kInt));
  VarHandle cV(c.data()->base_handle());

  auto* allocStmt = Allocate::make(cV, kInt, {2, 3, 4});
  auto* storeStmt = Store::make(c, {0, 0, 0}, 1, 1);
  auto* freeStmt = Free::make(cV);

  auto* block = Block::make({allocStmt, storeStmt, freeStmt});

  CostEstimator estimator;
  const Expr* cost = estimator.estimateCost(block);

  // Alloc cost is dependent on the amount of bytes allocated, but just check
  // its >0.
  ASSERT_TRUE(cost->isConstant());
  ASSERT_GT(immediateAs<float>(cost), 17);
}

void testCostEstimatorDictionary() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = ((ExprHandle(2)) * (x / y) * y) - ((x / y) * y);

  CostEstimator estimator1;
  const Expr* cost1 = estimator1.estimateCost(body.node());

  // Six binary ops.
  ASSERT_TRUE(cost1->isConstant());
  ASSERT_EQ(immediateAs<int>(cost1), 6);

  // Use a dictionary that makes referencing vars very expensive.
  OpCostDictionary dict;
  dict.VAR_REF_COST = 1000;

  CostEstimator estimator2(dict);
  const Expr* cost2 = estimator2.estimateCost(body.node());

  // Six binary ops and six expensive var references.
  ASSERT_TRUE(cost2->isConstant());
  ASSERT_EQ(immediateAs<int>(cost2), 6006);
}

void testCostEstimatorSanityCheck() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = ((ExprHandle(2)) * (x / y) * y) - ((x / y) * y);

  CostEstimator estimator;
  const Expr* cost1 = estimator.estimateCost(body.node());

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Expr* cost2 = estimator.estimateCost(simplified.node());

  // Check that simplification reduces estimated op cost for this expression
  // where we know it will.
  ASSERT_TRUE(cost1->isConstant());
  ASSERT_TRUE(cost2->isConstant());
  ASSERT_GT(immediateAs<int>(cost1), immediateAs<int>(cost2));
}

void testCostEstimatorFindCommonExpr() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  ExprHandle body = ((ExprHandle(2)) * (x / y) * y) - ((x / y) * y);

  Buffer c(BufHandle("C", {2}, kInt));
  auto* store1 = Store::make(c, {0}, body, 1);
  auto* store2 = Store::make(c, {1}, body, 1);

  auto* block = Block::make({store1, store2});

  CostEstimator estimator;
  auto info = estimator.getInfo(block);

  ASSERT_EQ(info.count, 1);

  // The body appears twice (in both stores).
  info = estimator.getInfo(body.node());
  ASSERT_EQ(info.count, 2);

  // TODO: the stores would also be count two because we don't have the
  // dependency info to know they're different. We should probably ensure all
  // Stores have different hashes.
}

void testCostEstimatorFindCommonParent() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle i("i", kInt);

  ExprHandle body = ((ExprHandle(2)) * (x / y) * y) - ((x / y) * y);

  Buffer c(BufHandle("C", {2}, kInt));
  auto* store1 = Store::make(c, {0}, body, 1);
  auto* for1 = For::make(i, 0, 5, store1);
  auto* store2 = Store::make(c, {1}, body, 1);
  auto* for2 = For::make(i, 0, 5, store2);

  auto* block = Block::make({for1, for2});

  CostEstimator estimator;
  auto info = estimator.getInfo(for1);
  ASSERT_EQ(info.count, 1);
  info = estimator.getInfo(body.node());
  ASSERT_EQ(info.parent, for1->body());

  estimator.clear();
  info = estimator.getInfo(block);

  // The body appears twice (in both stores).
  info = estimator.getInfo(body.node());
  ASSERT_EQ(info.count, 2);
  ASSERT_EQ(info.parent, block);

  // verify that if they have no shared parent we recognize that.
  estimator.clear();

  // don't care about info here but want to add the store's state.
  info = estimator.getInfo(store1);
  info = estimator.getInfo(store2);

  info = estimator.getInfo(body.node());
  ASSERT_EQ(info.count, 2);
  ASSERT_EQ(info.parent, nullptr);
}

} // namespace jit
} // namespace torch

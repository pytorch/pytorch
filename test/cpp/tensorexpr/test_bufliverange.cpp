#include <gtest/gtest.h>

#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>

#include <sstream>
namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

TEST(BufLiveRange, SingleRangeLine) {
  VarHandle i("i", kInt), j("j", kInt);
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32, 32}, kFloat);

  // Construct Stmt:
  // {
  //   for (int i = 0; i < 32; i++) {
  //     a[i] = 0;
  //     for (int j = 0; j < 32; j++) {
  //       a[i] = (a[i]) + (b[i, j]);
  //     }
  //   }
  // }

  StorePtr aInit = Store::make(a, {i}, 0);
  ExprHandle reduce = a.load({i}) + b.load({i, j});
  StorePtr aReduce = Store::make(a, {i}, reduce);
  StmtPtr loop =
      For::make(i, 0, 32, Block::make({aInit, For::make(j, 0, 32, aReduce)}));

  StmtPtr stmt = Block::make({loop});

  auto range = BufLiveRange::liveRange(stmt, a.node());
  ASSERT_TRUE(std::get<0>(range) == 0);
  ASSERT_TRUE(std::get<1>(range) == 0);
}

TEST(BufLiveRange, MulRangeLine) {
  VarHandle i("i", kInt);
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32}, kFloat);

  // Construct Stmt:
  // {
  //   for (int i = 0; i < 32; i++) {
  //     if (i<10 ? 1 : 0) {
  //       a[i] = i + i;
  //       b[i] = i * i;
  //     }
  //   }
  //   for (int i = 0; i < 32; i++) {
  //     if (i>10 ? 1 : 0) {
  //       a[i] = i * i;
  //       b[i] = i + i;
  //     }
  //   }
  // }

  StorePtr aStore_1 = Store::make(a, {i}, i + i);
  StorePtr bStore_1 = Store::make(b, {i}, i * i);
  StmtPtr loop_1 = For::make(
      i, 0, 32, Cond::make(i < 10, Block::make({aStore_1, bStore_1}), NULL));

  StorePtr aStore_2 = Store::make(a, {i}, i * i);
  StorePtr bStore_2 = Store::make(b, {i}, i + i);
  StmtPtr loop_2 = For::make(
      i, 0, 32, Cond::make(i > 10, Block::make({aStore_2, bStore_2}), NULL));

  StmtPtr stmt = Block::make({loop_1, loop_2});

  auto range_a = BufLiveRange::liveRange(stmt, a.node());
  ASSERT_TRUE(std::get<0>(range_a) == 0);
  ASSERT_TRUE(std::get<1>(range_a) == 1);

  auto range_b = BufLiveRange::liveRange(stmt, b.node());
  ASSERT_TRUE(std::get<0>(range_b) == 0);
  ASSERT_TRUE(std::get<1>(range_b) == 1);
}

} // namespace jit
} // namespace torch

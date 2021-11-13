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

TEST(StmtIndexer, BasicTest) {
  VarHandle i("i", kInt), j("j", kInt);
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32, 32}, kFloat);

  // Construct Stmt:
  // {
  //   for (int i = 0; i < 32; i++) {
  //     a[i] = 0; // StmtIndex: 0:0
  //     for (int j = 0; j < 32; j++) {
  //       a[i] = (a[i]) + (b[i, j]); // StmtIndex: 0:1:0
  //     }
  //   }
  // }

  StorePtr aInit = Store::make(a, {i}, 0);
  ExprHandle reduce = a.load({i}) + b.load({i, j});
  StorePtr aReduce = Store::make(a, {i}, reduce);
  StmtPtr loop =
      For::make(i, 0, 32, Block::make({aInit, For::make(j, 0, 32, aReduce)}));

  StmtPtr stmt = Block::make({loop});

  auto accesses = BufAccesses::find(stmt, a.node());
  int pass = 0;
  for (auto& acc : accesses) {
    auto store = std::get<0>(acc);
    auto stmt_index = std::get<2>(acc).getStmtIndexString();

    // check the stmtIndex of aInit
    if (store == aInit) {
      if (stmt_index == "0:0:") {
        pass += 1;
      }
    }

    // check the stmtIndex of aReduce
    if (store == aReduce) {
      if (stmt_index == "0:1:0:") {
        pass += 1;
      }
    }
  }

  ASSERT_TRUE(pass == 2);
}

TEST(StmtIndexer, CondTest) {
  VarHandle i("i", kInt);
  BufHandle a("a", {32}, kFloat);
  BufHandle b("b", {32}, kFloat);

  // Construct Stmt:
  // {
  //   for (int i = 0; i < 32; i++) {
  //     if (i<10 ? 1 : 0) {
  //       a[i] = i + i; // StmtIndex: 0:0:0
  //       b[i] = i * i; // StmtIndex: 0:0:1
  //     }
  //   }
  // }

  StorePtr aStore = Store::make(a, {i}, i + i);
  StorePtr bStore = Store::make(b, {i}, i * i);
  StmtPtr loop = For::make(
      i, 0, 32, Cond::make(i < 10, Block::make({aStore, bStore}), NULL));

  StmtPtr stmt = Block::make({loop});

  // check the stmtIndex of aStore
  auto acc_a = BufAccesses::find(stmt, a.node()).at(0);
  auto stmt_index_a = std::get<2>(acc_a).getStmtIndexString();
  ASSERT_TRUE(stmt_index_a == "0:0:0:");

  // check the stmtIndex of bStore
  auto acc_b = BufAccesses::find(stmt, b.node()).at(0);
  auto stmt_index_b = std::get<2>(acc_b).getStmtIndexString();
  ASSERT_TRUE(stmt_index_b == "0:0:1:");
}

} // namespace jit
} // namespace torch

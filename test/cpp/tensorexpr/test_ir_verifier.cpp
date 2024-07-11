#include <gtest/gtest.h>

#include <stdexcept>
#include "test/cpp/tensorexpr/test_base.h"

#include <torch/csrc/jit/tensorexpr/expr.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <sstream>
namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

TEST(IRVerifier, BitwiseOps) {
  VarPtr X = alloc<Var>("x", kInt);
  VarPtr Y = alloc<Var>("y", kFloat);
  {
    auto a = alloc<And>(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = alloc<Or>(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = alloc<Xor>(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = alloc<Lshift>(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = alloc<Rshift>(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, CompareSelect) {
  ExprPtr X = alloc<IntImm>(1);
  ExprPtr Y = alloc<FloatImm>(3.14f);
  {
    auto a = alloc<CompareSelect>(X, X, X, Y, kEQ);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = alloc<CompareSelect>(X, Y, X, X, kEQ);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, Ramp) {
  VarPtr I = alloc<Var>("i", kInt);
  VarPtr J = alloc<Var>("j", kFloat);
  {
    auto a = alloc<Ramp>(I, J, 4);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, Load) {
  VarPtr I = alloc<Var>("i", kInt);
  VarPtr J = alloc<Var>("j", kLong);
  VarPtr K = alloc<Var>("k", kFloat);
  BufPtr B = alloc<Buf>(
      "b",
      std::vector<ExprPtr>({alloc<IntImm>(10), alloc<IntImm>(20)}),
      kFloat);
  {
    // Indices with different int dtypes (kInt, kLong) are ok
    auto a = alloc<Load>(B, std::vector<ExprPtr>({I, J}));
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_NO_THROW(verify(a));
  }
  {
    // Float index
    auto a = alloc<Load>(B, std::vector<ExprPtr>({K, K}));
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Multilanes are only allowed in flattened indices
    auto multilane_index = alloc<Ramp>(I, alloc<IntImm>(1), 4);
    auto a = alloc<Load>(B, std::vector<ExprPtr>({I, multilane_index}));
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, IfThenElse) {
  VarPtr I = alloc<Var>("i", kInt);
  VarPtr J = alloc<Var>("j", kLong);
  VarPtr K = alloc<Var>("k", kFloat);
  {
    // Condition must be integral
    auto a = alloc<IfThenElse>(K, I, I);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Dtypes of true and false exprs must match
    auto a = alloc<IfThenElse>(I, I, J);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Can't have multiple lanes in condition expr
    auto a = alloc<IfThenElse>(alloc<Broadcast>(I, 4), I, I);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, For) {
  VarPtr I = alloc<Var>("i", kInt);
  VarPtr J = alloc<Var>("j", kInt);
  StmtPtr body = alloc<Block>(std::vector<StmtPtr>({}));
  {
    // Can't have nullptr as a Var
    auto a = alloc<For>(nullptr, I, J, body);
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    EXPECT_ANY_THROW(verify(a));
  }
}

TEST(IRVerifier, Block) {
  VarPtr I = alloc<Var>("i", kInt);
  BufPtr B = alloc<Buf>("B", std::vector<ExprPtr>({alloc<IntImm>(10)}), kInt);
  {
    StmtPtr store = alloc<Store>(B, std::vector<ExprPtr>({I}), I);
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    StmtPtr block1 = alloc<Block>(std::vector<StmtPtr>({store}));
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    StmtPtr block2 = alloc<Block>(std::vector<StmtPtr>({store}));
    // Stmt can't have multiple parents, thus inserting it into several blocks
    // is illegal
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(block2));
  }
}

TEST(IRVerifier, Store) {
  VarPtr I = alloc<Var>("i", kInt);
  VarPtr J = alloc<Var>("j", kLong);
  VarPtr K = alloc<Var>("k", kFloat);
  BufPtr B = alloc<Buf>(
      "b",
      std::vector<ExprPtr>({alloc<IntImm>(10), alloc<IntImm>(20)}),
      kFloat);
  {
    // Indices with different int dtypes (kInt, kLong) are ok
    auto a = alloc<Store>(B, std::vector<ExprPtr>({I, J}), K);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_NO_THROW(verify(a));
  }
  {
    // Float index
    auto a = alloc<Store>(B, std::vector<ExprPtr>({K, K}), K);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Multilanes are only allowed in flattened indices
    auto multilane_index = alloc<Ramp>(I, alloc<IntImm>(1), 4);
    auto a = alloc<Store>(B, std::vector<ExprPtr>({I, multilane_index}), K);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Value and buf dtypes mismatch
    auto a = alloc<Store>(B, std::vector<ExprPtr>({I}), I);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

} // namespace jit
} // namespace torch

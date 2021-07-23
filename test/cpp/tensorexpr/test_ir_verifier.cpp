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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, BitwiseOps) {
  KernelScope kernel_scope;
  const Var* X = new Var("x", kInt);
  const Var* Y = new Var("y", kFloat);
  {
    auto a = new And(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = new Or(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = new Xor(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = new Lshift(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = new Rshift(X, Y);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, CompareSelect) {
  KernelScope kernel_scope;
  const Expr* X = new IntImm(1);
  const Expr* Y = new FloatImm(3.14f);
  {
    auto a = new CompareSelect(X, X, X, Y, kEQ);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    auto a = new CompareSelect(X, Y, X, X, kEQ);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, Ramp) {
  KernelScope kernel_scope;
  const Var* I = new Var("i", kInt);
  const Var* J = new Var("j", kFloat);
  {
    auto a = new Ramp(I, J, 4);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, Load) {
  KernelScope kernel_scope;
  const Var* I = new Var("i", kInt);
  const Var* J = new Var("j", kLong);
  const Var* K = new Var("k", kFloat);
  const Buf* B = new Buf("b", {new IntImm(10), new IntImm(20)}, kFloat);
  {
    // Indices with different int dtypes (kInt, kLong) are ok
    auto a = new Load(B, {I, J});
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_NO_THROW(verify(a));
  }
  {
    // Float index
    auto a = new Load(B, {K, K});
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Multilanes are only allowed in flattened indices
    auto multilane_index = new Ramp(I, new IntImm(1), 4);
    auto a = new Load(B, {I, multilane_index});
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, IfThenElse) {
  KernelScope kernel_scope;
  const Var* I = new Var("i", kInt);
  const Var* J = new Var("j", kLong);
  const Var* K = new Var("k", kFloat);
  {
    // Condition must be integral
    auto a = new IfThenElse(K, I, I);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Dtypes of true and false exprs must match
    auto a = new IfThenElse(I, I, J);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Can't have multiple lanes in condition expr
    auto a = new IfThenElse(new Broadcast(I, 4), I, I);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, For) {
  KernelScope kernel_scope;
  const Var* I = new Var("i", kInt);
  const Var* J = new Var("j", kInt);
  Stmt* body = new Block({});
  {
    // Can't have nullptr as a Var
    auto a = new For(nullptr, I, J, body);
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    EXPECT_ANY_THROW(verify(a));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, Block) {
  KernelScope kernel_scope;
  const Var* I = new Var("i", kInt);
  const Buf* B = new Buf("B", {new IntImm(10)}, kInt);
  {
    Stmt* store = new Store(B, {I}, I);
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    Stmt* block1 = new Block({store});
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    Stmt* block2 = new Block({store});
    // Stmt can't have multiple parrents, thus inserting it into several blocks
    // is illegal
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(block2));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(IRVerifier, Store) {
  KernelScope kernel_scope;
  const Var* I = new Var("i", kInt);
  const Var* J = new Var("j", kLong);
  const Var* K = new Var("k", kFloat);
  const Buf* B = new Buf("b", {new IntImm(10), new IntImm(20)}, kFloat);
  {
    // Indices with different int dtypes (kInt, kLong) are ok
    auto a = new Store(B, {I, J}, K);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_NO_THROW(verify(a));
  }
  {
    // Float index
    auto a = new Store(B, {K, K}, K);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Multilanes are only allowed in flattened indices
    auto multilane_index = new Ramp(I, new IntImm(1), 4);
    auto a = new Store(B, {I, multilane_index}, K);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
  {
    // Value and buf dtypes mismatch
    auto a = new Store(B, {I}, I);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto,clang-analyzer-cplusplus.NewDeleteLeaks)
    EXPECT_ANY_THROW(verify(a));
  }
}

} // namespace jit
} // namespace torch

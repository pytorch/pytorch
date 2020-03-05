#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/constant_folder.h"

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;

void testConstantFoldSimple() {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle f = (a + b);

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(f.node()->accept_mutator(&folder));
  EXPECT_NE(newF.AsNode<FloatImm>(), nullptr);
  EXPECT_EQ(newF.AsNode<FloatImm>()->value(), 5);

  SimpleIRExprEval eval(newF);
  EXPECT_EQ(eval.value<float>(), 5.f);
}

void testConstantFoldTwoLayer() {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle f = (a + b) - (c + d);

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(f.node()->accept_mutator(&folder));
  EXPECT_NE(newF.AsNode<FloatImm>(), nullptr);
  EXPECT_EQ(newF.AsNode<FloatImm>()->value(), -4);

  SimpleIRExprEval eval(newF);
  EXPECT_EQ(eval.value<float>(), -4.f);
}

void testConstantFoldShifts() {
  KernelScope kernel_scope;
  ExprHandle a(7);
  ExprHandle b(2);
  ExprHandle c(3);
  ExprHandle f = ((a << b) << b) >> c;

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(f.node()->accept_mutator(&folder));
  EXPECT_NE(newF.AsNode<IntImm>(), nullptr);
  EXPECT_EQ(newF.AsNode<IntImm>()->value(), 14);

  SimpleIRExprEval eval(newF);
  EXPECT_EQ(eval.value<int>(), 7 << (4-3));
}

void testConstantFoldBitwise() {
  KernelScope kernel_scope;
  ExprHandle a(59);
  ExprHandle b(22);
  ExprHandle c(101);
  ExprHandle f = (a ^ b) & c;

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(f.node()->accept_mutator(&folder));
  EXPECT_NE(newF.AsNode<IntImm>(), nullptr);
  EXPECT_EQ(newF.AsNode<IntImm>()->value(), 37);

  SimpleIRExprEval eval(newF);
  EXPECT_EQ(eval.value<int>(), (59 ^ 22) & 101);
}

void testConstantFoldMultiOp() {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle e(6.0f);
  ExprHandle f(7.0f);
  ExprHandle fn = ((a / e) - (c + d)) * (f / b);

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(fn.node()->accept_mutator(&folder));
  EXPECT_NE(newF.AsNode<FloatImm>(), nullptr);

  SimpleIRExprEval eval(newF);
  SimpleIRExprEval ref(fn);

  EXPECT_EQ(eval.value<float>(), ref.value<float>());
}

void testConstantFoldMinMax() {
  KernelScope kernel_scope;
  ExprHandle a(12.0f);
  ExprHandle b(15.0f);
  ExprHandle c(17.0f);

  // x = max(12, min(15, 17)).
  ExprHandle minHandle = Min::make(b, c, true);
  ExprHandle fn = Max::make(a, minHandle, false);

  EXPECT_EQ(fn.dtype().scalar_type(), ScalarType::Float);

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(fn.node()->accept_mutator(&folder));
  EXPECT_NE(newF.AsNode<FloatImm>(), nullptr);

  SimpleIRExprEval eval(newF);
  EXPECT_EQ(eval.value<float>(), 15.f);
}

void testConstantFoldIntrinsics() {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle powHandle = Intrinsics::make(kPow, a, b);
  ExprHandle sinHandle = Intrinsics::make(kSin, powHandle);
  ExprHandle modHandle = Intrinsics::make(kFmod, c, sinHandle);
  ExprHandle logHandle = Intrinsics::make(kLog10, modHandle);
  ExprHandle rndHandle = Intrinsics::make(kRound, logHandle);
  ExprHandle fn = Intrinsics::make(kFabs, rndHandle);

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(fn.node()->accept_mutator(&folder));
  EXPECT_NE(newF.AsNode<FloatImm>(), nullptr);
  EXPECT_EQ(newF.AsNode<FloatImm>()->value(), 1);

  SimpleIRExprEval eval(newF);
  SimpleIRExprEval ref(fn);

  EXPECT_EQ(eval.value<float>(), ref.value<float>());
}

void testConstantFoldWithVar() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle body = x * (ExprHandle(2.f) + ExprHandle(4.f));

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(body.node()->accept_mutator(&folder));
  const Mul* root = newF.AsNode<Mul>();
  EXPECT_NE(root, nullptr);
  EXPECT_NE(dynamic_cast<const FloatImm*>(root->rhs()), nullptr);

  ExprHandle result = Let::make(x, ExprHandle(3.f), newF);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<float>(), 3 * (2 + 4));
}

void testUnFoldableExpr() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle body = (ExprHandle(3) * x) + (ExprHandle(5) * y);

  ConstantFolder folder;
  ExprHandle newF = ExprHandle(body.node()->accept_mutator(&folder));
  const Add* root = newF.AsNode<Add>();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(dynamic_cast<const FloatImm*>(root->lhs()), nullptr);
  EXPECT_EQ(dynamic_cast<const FloatImm*>(root->rhs()), nullptr);

  ExprHandle result = Let::make(x, ExprHandle(3.f), newF);
  result = Let::make(y, ExprHandle(2.f), result);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<float>(), 9 + 10);
}

} // namespace jit
} // namespace torch

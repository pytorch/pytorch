#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/hash_server.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;

void testConstantFoldSimple() {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle f = (a + b);

  ExprHandle newF = IRSimplifier::simplify(f);
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

  ExprHandle newF = IRSimplifier::simplify(f);
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

  ExprHandle newF = IRSimplifier::simplify(f);
  EXPECT_NE(newF.AsNode<IntImm>(), nullptr);
  EXPECT_EQ(newF.AsNode<IntImm>()->value(), 14);

  SimpleIRExprEval eval(newF);
  EXPECT_EQ(eval.value<int>(), 7 << (4 - 3));
}

void testConstantFoldBitwise() {
  KernelScope kernel_scope;
  ExprHandle a(59);
  ExprHandle b(22);
  ExprHandle c(101);
  ExprHandle f = (a ^ b) & c;

  ExprHandle newF = IRSimplifier::simplify(f);
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

  ExprHandle newF = IRSimplifier::simplify(fn);
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

  ExprHandle newF = IRSimplifier::simplify(fn);
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

  ExprHandle newF = IRSimplifier::simplify(fn);
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

  ExprHandle newF = IRSimplifier::simplify(body);
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

  ExprHandle newF = IRSimplifier::simplify(body);
  const Add* root = newF.AsNode<Add>();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(dynamic_cast<const FloatImm*>(root->lhs()), nullptr);
  EXPECT_EQ(dynamic_cast<const FloatImm*>(root->rhs()), nullptr);

  ExprHandle result = Let::make(x, ExprHandle(3.f), newF);
  result = Let::make(y, ExprHandle(2.f), result);
  SimpleIRExprEval eval(result);
  EXPECT_EQ(eval.value<float>(), 9 + 10);
}

void testHashSimple() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle f = a + b * x;

  HashProvider hasher;

  auto hash_x = hasher.hash(x.node());
  auto hash_a = hasher.hash(a.node());
  auto hash_f = hasher.hash(f.node());

  EXPECT_NE(hash_x, 0);
  EXPECT_NE(hash_a, 0);
  EXPECT_NE(hash_f, 0);
  EXPECT_NE(hash_x, hash_a);
  EXPECT_NE(hash_x, hash_f);
  EXPECT_NE(hash_a, hash_f);
}

void testHashEquivalence() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle f = (x * y) + (x * y);

  const Add* root = f.AsNode<Add>();
  EXPECT_NE(root, nullptr);

  HashProvider hasher;
  auto hash_f = hasher.hash(f.node());
  auto hash_l = hasher.hash(root->lhs());
  auto hash_r = hasher.hash(root->rhs());

  // Root not equal to either branch.
  EXPECT_NE(hash_f, hash_l);
  EXPECT_NE(hash_f, hash_r);
  // but branches are equal.
  EXPECT_EQ(hash_l, hash_r);

  // Still equivalent if separate.
  ExprHandle a(2);
  ExprHandle f2 = x + a / y;
  ExprHandle b(2);
  ExprHandle f3 = x + b / y;
  EXPECT_EQ(hasher.hash(f2.node()), hasher.hash(f3.node()));

  // Not equivalent if different vars (even with same name).
  VarHandle z("x", kFloat);
  ExprHandle f4 = z + b / y;
  EXPECT_NE(hasher.hash(f2.node()), hasher.hash(f4.node()));

  // Intrinsics sanity check.
  ExprHandle f5 = Intrinsics::make(kSin, x) * Intrinsics::make(kCos, x);
  EXPECT_NE(hasher.hash(f5.node()), 0);
}

void testHashEquivalenceAfterFolding() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(5.0f);
  ExprHandle f = ((a + b) * x) * (c * x);

  const Mul* root = f.AsNode<Mul>();
  EXPECT_NE(root, nullptr);

  HashProvider hasher;
  auto hash_f = hasher.hash(f.node());
  auto hash_l = hasher.hash(root->lhs());
  auto hash_r = hasher.hash(root->rhs());

  // Root not equal to either branch, and branches not equal.
  EXPECT_NE(hash_f, hash_l);
  EXPECT_NE(hash_f, hash_r);
  EXPECT_NE(hash_l, hash_r);

  ExprHandle newF = IRSimplifier::simplify(f);

  const Mul* newRoot = newF.AsNode<Mul>();
  EXPECT_NE(newRoot, nullptr);

  auto hash_f_n = hasher.hash(newF.node());
  auto hash_l_n = hasher.hash(newRoot->lhs());
  auto hash_r_n = hasher.hash(newRoot->rhs());

  // Root not equal to either branch.
  EXPECT_NE(hash_f_n, hash_l_n);
  EXPECT_NE(hash_f_n, hash_r_n);
  // but branches are now equal.
  EXPECT_EQ(hash_l_n, hash_r_n);
}

void testHashDifferenceTypes() {
  KernelScope kernel_scope;

  HashProvider hasher;
  std::vector<const Expr*> immediates;

  immediates.push_back(new DoubleImm(1));
  immediates.push_back(new FloatImm(1));
  immediates.push_back(new HalfImm(1));
  immediates.push_back(new BoolImm(1));
  immediates.push_back(new CharImm(1));
  immediates.push_back(new ByteImm(1));
  immediates.push_back(new ShortImm(1));
  immediates.push_back(new IntImm(1));
  immediates.push_back(new LongImm(1));

  // Immediates of different types are not equal.
  for (unsigned int i = 0; i < immediates.size(); ++i) {
    for (unsigned int j = i + 1; j < immediates.size(); ++j) {
      EXPECT_NE(hasher.hash(immediates[i]), hasher.hash(immediates[j]));
    }
  }

  // But coerced immediates are if they are the same type:
  ExprHandle f1 = ExprHandle(2.f) + CharImm::make(1);
  ExprHandle f2 = Cast::make(kFloat, IntImm::make(3));

  ExprHandle ff1 = IRSimplifier::simplify(f1);
  ExprHandle ff2 = IRSimplifier::simplify(f2);

  EXPECT_EQ(hasher.hash(ff1.node()), hasher.hash(ff2.node()));
}

void testHashLargeExpression() {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Buffer a(VarHandle("A", kHandle), kInt, {N});
  Buffer b(VarHandle("B", kHandle), kInt, {N});
  Buffer c(VarHandle("C", kHandle), kInt, {N});
  auto mask = IntImm::make(1);
  VarHandle i("i", kInt);
  auto memcpy_stmt = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kEQ),
          mask));

  Buffer d(VarHandle("D", kHandle), kInt, {1});
  Buffer e(VarHandle("E", kHandle), kInt, {1});
  auto store_ramp_stmt = Store::make(
      e,
      Ramp::make(0, 1, 4),
      Load::make(d, Ramp::make(0, 1, 4), Broadcast::make(IntImm::make(1), 4)),
      Broadcast::make(Cast::make(kInt, DoubleImm::make(1)), 4));

  auto if_stmt = Cond::make(
      CompareSelect::make(
          Load::make(a, i, mask),
          Load::make(b, i, mask),
          CompareSelectOperation::kGE),
      memcpy_stmt,
      store_ramp_stmt);

  HashProvider hasher;
  auto hash_r = hasher.hash(if_stmt);
  // We should not have to do any more work.
  EXPECT_TRUE(hasher.cachedHash(memcpy_stmt));
  auto hash_t = hasher.hash(memcpy_stmt);
  EXPECT_TRUE(hasher.cachedHash(store_ramp_stmt));
  auto hash_f = hasher.hash(store_ramp_stmt);

  // Root not equal to either branch, and branches not equal.
  EXPECT_NE(hash_r, hash_t);
  EXPECT_NE(hash_r, hash_f);
  EXPECT_NE(hash_t, hash_f);
}

/// (2.f + x) + 4.f => x + 6.f
void testSimplifyAdd() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle body = (ExprHandle(2.f) + x) + ExprHandle(4.f);

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Add* root = simplified.AsNode<Add>();
  EXPECT_NE(root, nullptr);
  const Var* lhs = dynamic_cast<const Var*>(root->lhs());
  EXPECT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->name_hint(), "x");
  const FloatImm* rhs = dynamic_cast<const FloatImm*>(root->rhs());
  EXPECT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->value(), 6.f);
}

/// (2.f - x) - 4.f => -2.f - x
void testSimplifySub() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle body = (ExprHandle(2.f) - x) - ExprHandle(4.f);

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Sub* root = simplified.AsNode<Sub>();
  EXPECT_NE(root, nullptr);
  const FloatImm* lhs = dynamic_cast<const FloatImm*>(root->lhs());
  EXPECT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->value(), -2.f);
  const Var* rhs = dynamic_cast<const Var*>(root->rhs());
  EXPECT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->name_hint(), "x");
}

/// 2.f * (1.f - x) - 4.f => -6.f - (x * 2.f)
void testSimplifyMultiLayer() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle body = ExprHandle(2.f) * ((ExprHandle(1.f) - x) - ExprHandle(4.f));

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Sub* root = simplified.AsNode<Sub>();
  EXPECT_NE(root, nullptr);
  const FloatImm* lhs = dynamic_cast<const FloatImm*>(root->lhs());
  EXPECT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->value(), -6.f);
  const Mul* rhs = dynamic_cast<const Mul*>(root->rhs());
  EXPECT_NE(rhs, nullptr);
  const Var* varX = dynamic_cast<const Var*>(rhs->lhs());
  EXPECT_NE(varX, nullptr);
  EXPECT_EQ(varX->name_hint(), "x");
  const FloatImm* mulRhs = dynamic_cast<const FloatImm*>(rhs->rhs());
  EXPECT_NE(mulRhs, nullptr);
  EXPECT_EQ(mulRhs->value(), 2.f);
}

/// 2 * (3 * x) - (x * 4) => x * 2
void testSimplifyMultiTerm() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body =
      (ExprHandle(2) * ((ExprHandle(3) * x)) - (x * ExprHandle(4)));

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Mul* root = simplified.AsNode<Mul>();
  EXPECT_NE(root, nullptr);
  const Var* lhs = dynamic_cast<const Var*>(root->lhs());
  EXPECT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->name_hint(), "x");
  const IntImm* rhs = dynamic_cast<const IntImm*>(root->rhs());
  EXPECT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->value(), 2);
}

/// 2 * (3 * (f)x) - (x * 4) => x * 2.f
void testSimplifyCasts() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle body =
      (ExprHandle(2) * ((ExprHandle(3) * x)) - (x * ExprHandle(4)));

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Mul* root = simplified.AsNode<Mul>();
  EXPECT_NE(root, nullptr);
  const Var* lhs = dynamic_cast<const Var*>(root->lhs());
  EXPECT_NE(lhs, nullptr);
  EXPECT_EQ(lhs->name_hint(), "x");
  const FloatImm* rhs = dynamic_cast<const FloatImm*>(root->rhs());
  EXPECT_NE(rhs, nullptr);
  EXPECT_EQ(rhs->value(), 2);
}

/// (x + 0) * 1 => x
void testSimplifyEliminatesNoOps() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body = (x + ExprHandle(0)) * 1;

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Var* root = simplified.AsNode<Var>();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(root->name_hint(), "x");
}

/// Cannot simplify this.
void testSimplifyMultiVar() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = y * 24 + x * 34;

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Add* root = simplified.AsNode<Add>();
  EXPECT_NE(root, nullptr);
  const Mul* lhs = dynamic_cast<const Mul*>(root->lhs());
  EXPECT_NE(lhs, nullptr);
  const Var* varY = dynamic_cast<const Var*>(lhs->lhs());
  EXPECT_EQ(varY->name_hint(), "y");
  const Mul* rhs = dynamic_cast<const Mul*>(root->rhs());
  EXPECT_NE(rhs, nullptr);
  const Var* varX = dynamic_cast<const Var*>(rhs->lhs());
  EXPECT_NE(varX, nullptr);
  EXPECT_EQ(varX->name_hint(), "x");
}

/// y + x * 0 => y
void testSimplifyEliminatesVar() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = y + x * ExprHandle(0);

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Var* root = simplified.AsNode<Var>();
  EXPECT_NE(root, nullptr);
  EXPECT_EQ(root->name_hint(), "y");
}

} // namespace jit
} // namespace torch

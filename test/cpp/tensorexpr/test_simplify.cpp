#include <gtest/gtest.h>
#include <test/cpp/tensorexpr/test_base.h>

#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

#include <cmath>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;
using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;

#define IS_NODE(T, node)                        \
  {                                             \
    auto* node_ = dynamic_cast<const T*>(node); \
    ASSERT_NE(nullptr, node_);                  \
  }

#define IS_NODE_WITH_NAME(T, node, name)     \
  auto* name = dynamic_cast<const T*>(node); \
  ASSERT_NE(nullptr, name);

#define IS_NODE_WITH_NAME_AND_CAST(T, node, name, Type)        \
  const T* name = nullptr;                                     \
  {                                                            \
    auto* node_ = dynamic_cast<const Cast*>(node);             \
    ASSERT_NE(nullptr, node_);                                 \
    ASSERT_EQ(node_->dtype().scalar_type(), ScalarType::Type); \
    name = dynamic_cast<const T*>(node_->src_value());         \
  }                                                            \
  ASSERT_NE(nullptr, name);

#define IS_IMM_WITH_VAL(T, node, val)                \
  {                                                  \
    auto* node_ = dynamic_cast<const T##Imm*>(node); \
    ASSERT_NE(nullptr, node_);                       \
    ASSERT_EQ(node_->value(), val);                  \
  }

#define IS_VAR_WITH_NAME(node, name)              \
  {                                               \
    auto* node_ = dynamic_cast<const Var*>(node); \
    ASSERT_NE(nullptr, node_);                    \
    ASSERT_EQ(node_->name_hint(), name);          \
  }

#define IS_BINOP_W_VARS(T, node, name, v1, v2) \
  const T* name = nullptr;                     \
  {                                            \
    name = dynamic_cast<const T*>(node);       \
    ASSERT_NE(nullptr, name);                  \
    IS_VAR_WITH_NAME(name->lhs(), v1);         \
    IS_VAR_WITH_NAME(name->rhs(), v2);         \
  }

#define IS_BINOP_W_CONST(T, node, name, v, c) \
  const T* name = nullptr;                    \
  {                                           \
    name = dynamic_cast<const T*>(node);      \
    ASSERT_NE(nullptr, name);                 \
    IS_VAR_WITH_NAME(name->lhs(), v);         \
    IS_IMM_WITH_VAL(Int, name->rhs(), c);     \
  }

#define IS_RAND(node)                                    \
  {                                                      \
    auto* node_ = dynamic_cast<const Intrinsics*>(node); \
    ASSERT_NE(nullptr, node_);                           \
    ASSERT_EQ(node_->op_type(), kRand);                  \
  }

TEST(Simplify, ConstantFoldSimple) {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle f = (a + b);

  ExprHandle newF = IRSimplifier::simplify(f);
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);
  ASSERT_EQ(newF.AsNode<FloatImm>()->value(), 5);

  SimpleIRExprEval eval(newF);
  ASSERT_EQ(eval.value<float>(), 5.f);
}

TEST(Simplify, ConstantFoldTwoLayer) {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle f = (a + b) - (c + d);

  ExprHandle newF = IRSimplifier::simplify(f);
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);
  ASSERT_EQ(newF.AsNode<FloatImm>()->value(), -4);

  SimpleIRExprEval eval(newF);
  ASSERT_EQ(eval.value<float>(), -4.f);
}

TEST(Simplify, ConstantFoldShifts) {
  KernelScope kernel_scope;
  ExprHandle a(7);
  ExprHandle b(2);
  ExprHandle c(3);
  ExprHandle f = ((a << b) << b) >> c;

  ExprHandle newF = IRSimplifier::simplify(f);
  ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
  ASSERT_EQ(newF.AsNode<IntImm>()->value(), 14);

  SimpleIRExprEval eval(newF);
  ASSERT_EQ(eval.value<int>(), 7 << (4 - 3));
}

TEST(Simplify, ConstantFoldBitwise) {
  KernelScope kernel_scope;
  ExprHandle a(59);
  ExprHandle b(22);
  ExprHandle c(101);
  ExprHandle f = (a ^ b) & c;

  ExprHandle newF = IRSimplifier::simplify(f);
  ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
  ASSERT_EQ(newF.AsNode<IntImm>()->value(), 37);

  SimpleIRExprEval eval(newF);
  ASSERT_EQ(eval.value<int>(), (59 ^ 22) & 101);
}

TEST(Simplify, ConstantFoldMultiOp) {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle e(6.0f);
  ExprHandle f(7.0f);
  ExprHandle fn = ((a / e) - (c + d)) * (f / b);

  ExprHandle newF = IRSimplifier::simplify(fn);
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);

  SimpleIRExprEval eval(newF);
  SimpleIRExprEval ref(fn);

  ASSERT_EQ(eval.value<float>(), ref.value<float>());
}

TEST(Simplify, ConstantFoldMinMax) {
  KernelScope kernel_scope;
  ExprHandle a(12.0f);
  ExprHandle b(15.0f);
  ExprHandle c(17.0f);

  // x = max(12, min(15, 17)).
  ExprHandle minHandle = Min::make(b, c, true);
  ExprHandle fn = Max::make(a, minHandle, false);

  ASSERT_EQ(fn.dtype().scalar_type(), ScalarType::Float);

  ExprHandle newF = IRSimplifier::simplify(fn);
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);

  SimpleIRExprEval eval(newF);
  ASSERT_EQ(eval.value<float>(), 15.f);
}

TEST(Simplify, ConstantFoldIntrinsics) {
  KernelScope kernel_scope;
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle powHandle = Intrinsics::make(kPow, a, b);
  ExprHandle sinHandle = Intrinsics::make(kSin, powHandle);
  ExprHandle modHandle = Intrinsics::make(kFmod, c, sinHandle);
  ExprHandle logHandle = Intrinsics::make(kLog10, modHandle);
  ExprHandle rndHandle = Intrinsics::make(kRound, logHandle);
  ExprHandle fn = Intrinsics::make(kAbs, rndHandle);

  ExprHandle newF = IRSimplifier::simplify(fn);
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);
  ASSERT_EQ(newF.AsNode<FloatImm>()->value(), 1);

  SimpleIRExprEval eval(newF);
  SimpleIRExprEval ref(fn);

  ASSERT_EQ(eval.value<float>(), ref.value<float>());
}

TEST(Simplify, ConstantFoldCastToBool) {
  KernelScope kernel_scope;
  ExprHandle f = Cast::make(kBool, IntImm::make(0));
  ExprHandle newF = IRSimplifier::simplify(f);
  SimpleIRExprEval eval(newF);
  ASSERT_EQ(eval.value<bool>(), false);
}

TEST(Simplify, ConstantFoldWithVar) {
  KernelScope kernel_scope;
  {
    VarHandle x("x", kInt);
    ExprHandle body = x * (ExprHandle(2) + ExprHandle(4));

    ExprHandle newF = IRSimplifier::simplify(body);
    const Mul* root = newF.AsNode<Mul>();
    ASSERT_NE(root, nullptr);
    ASSERT_NE(dynamic_cast<const IntImm*>(root->lhs()), nullptr);

    SimpleIRExprEval eval(newF);
    eval.bindVar(x, ExprHandle(3));
    ASSERT_EQ(eval.value<int>(), 3 * (2 + 4));
  }

  {
    VarHandle x("x", kFloat);
    ExprHandle body = x * (ExprHandle(2.f) + ExprHandle(4.f));

    ExprHandle newF = IRSimplifier::simplify(body);
    const Mul* root = newF.AsNode<Mul>();
    ASSERT_NE(root, nullptr);
    ASSERT_NE(dynamic_cast<const FloatImm*>(root->rhs()), nullptr);

    SimpleIRExprEval eval(newF);
    eval.bindVar(x, ExprHandle(3.f));
    ASSERT_EQ(eval.value<float>(), 3 * (2 + 4));
  }
}

TEST(Simplify, ConditionalSelectFoldSimple) {
  KernelScope kernel_scope;
  ExprHandle a(3.0f);
  ExprHandle b(4.0f);
  ExprHandle c(3.0f);
  {
    ExprHandle f = (a > b);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 0);
  }
  {
    ExprHandle f = (a < b);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    ExprHandle f = (a == c);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    ExprHandle f = (a != c);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 0);
  }
}

TEST(Simplify, ConditionalSelectFoldTwoLayer) {
  KernelScope kernel_scope;
  ExprHandle a(3.0f);
  ExprHandle b(2.0f);
  ExprHandle c(2.0f);
  ExprHandle d(1.0f);
  {
    ExprHandle f = (a + b < c + d);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 0);
  }
  {
    ExprHandle f = (a + b > c + d);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    ExprHandle f = (a + d == b + c);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 1);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    ExprHandle f = (a + d != b + c);

    ExprHandle newF = IRSimplifier::simplify(f);
    ASSERT_NE(newF.AsNode<IntImm>(), nullptr);
    ASSERT_EQ(newF.AsNode<IntImm>()->value(), 0);

    SimpleIRExprEval eval(newF);
    ASSERT_EQ(eval.value<int>(), 0);
  }
}

TEST(Simplify, ConditionalSelectFoldWithVar) {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle f = x < 4.f;

  ExprHandle newF = IRSimplifier::simplify(f);
  const IntImm* folded = newF.AsNode<IntImm>();
  ASSERT_EQ(folded, nullptr);

  {
    SimpleIRExprEval eval(newF);
    eval.bindVar(x, ExprHandle(3.f));
    ASSERT_EQ(eval.value<int>(), 1);
  }
  {
    SimpleIRExprEval eval(newF);
    eval.bindVar(x, ExprHandle(5.f));
    ASSERT_EQ(eval.value<int>(), 0);
  }
}

TEST(Simplify, UnFoldableExpr) {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle body = (ExprHandle(3) * x) + (ExprHandle(5) * y);

  ExprHandle newF = IRSimplifier::simplify(body);
  const Add* root = newF.AsNode<Add>();
  ASSERT_NE(root, nullptr);
  ASSERT_EQ(dynamic_cast<const FloatImm*>(root->lhs()), nullptr);
  ASSERT_EQ(dynamic_cast<const FloatImm*>(root->rhs()), nullptr);

  SimpleIRExprEval eval(newF);
  eval.bindVar(x, ExprHandle(3.f));
  eval.bindVar(y, ExprHandle(2.f));
  ASSERT_EQ(eval.value<float>(), 9 + 10);
}

TEST(Simplify, HashSimple) {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle f = a + b * x;

  HashProvider hasher;

  auto hash_x = hasher.hash(x.node());
  auto hash_a = hasher.hash(a.node());
  auto hash_f = hasher.hash(f.node());

  ASSERT_NE(hash_x, (size_t)0);
  ASSERT_NE(hash_a, (size_t)0);
  ASSERT_NE(hash_f, (size_t)0);
  ASSERT_NE(hash_x, hash_a);
  ASSERT_NE(hash_x, hash_f);
  ASSERT_NE(hash_a, hash_f);
}

TEST(Simplify, HashEquivalence) {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle f = (x * y) + (x * y);

  const Add* root = f.AsNode<Add>();
  ASSERT_NE(root, nullptr);

  HashProvider hasher;
  auto hash_f = hasher.hash(f.node());
  auto hash_l = hasher.hash(root->lhs());
  auto hash_r = hasher.hash(root->rhs());

  // Root not equal to either branch.
  ASSERT_NE(hash_f, hash_l);
  ASSERT_NE(hash_f, hash_r);
  // but branches are equal.
  ASSERT_EQ(hash_l, hash_r);

  // Still equivalent if separate.
  ExprHandle a(2);
  ExprHandle f2 = x + a / y;
  ExprHandle b(2);
  ExprHandle f3 = x + b / y;
  ASSERT_EQ(hasher.hash(f2.node()), hasher.hash(f3.node()));

  // Not equivalent if different vars (even with same name).
  VarHandle z("x", kFloat);
  ExprHandle f4 = z + b / y;
  ASSERT_NE(hasher.hash(f2.node()), hasher.hash(f4.node()));

  // Intrinsics sanity check.
  ExprHandle f5 = Intrinsics::make(kSin, x) * Intrinsics::make(kCos, x);
  ASSERT_NE(hasher.hash(f5.node()), (size_t)0);
}

TEST(Simplify, HashEquivalenceRand) {
  KernelScope kernel_scope;
  ExprHandle f =
      Intrinsics::make(kRand, kFloat) + Intrinsics::make(kRand, kInt);

  const Add* root = f.AsNode<Add>();
  ASSERT_NE(root, nullptr);

  HashProvider hasher;
  auto hash_f = hasher.hash(f.node());
  auto hash_l = hasher.hash(root->lhs());
  auto hash_r = hasher.hash(root->rhs());

  // Root not equal to either branch.
  ASSERT_NE(hash_f, hash_l);
  ASSERT_NE(hash_f, hash_r);
  // and branches are NOT equal.
  ASSERT_NE(hash_l, hash_r);
}

TEST(Simplify, HashEquivalenceAfterFolding) {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(5.0f);

  ExprHandle f1 = ((a + b) * x);
  ExprHandle f2 = (c * x);

  HashProvider hasher;
  auto hash_l = hasher.hash(f1.node());
  auto hash_r = hasher.hash(f2.node());

  // Root not equal to either branch, and branches not equal.
  ASSERT_NE(hash_l, hash_r);

  ExprHandle ff1 = IRSimplifier::simplify(f1);
  ExprHandle ff2 = IRSimplifier::simplify(f2);

  auto hash_l_n = hasher.hash(ff1.node());
  auto hash_r_n = hasher.hash(ff2.node());
  // but branches are now equal.
  ASSERT_EQ(hash_l_n, hash_r_n);
}

TEST(Simplify, HashDifferenceTypes) {
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
      ASSERT_NE(hasher.hash(immediates[i]), hasher.hash(immediates[j]));
    }
  }

  // But coerced immediates are if they are the same type:
  ExprHandle f1 = ExprHandle(2.f) + CharImm::make(1);
  ExprHandle f2 = Cast::make(kFloat, IntImm::make(3));

  ExprHandle ff1 = IRSimplifier::simplify(f1);
  ExprHandle ff2 = IRSimplifier::simplify(f2);

  ASSERT_EQ(hasher.hash(ff1.node()), hasher.hash(ff2.node()));
}

TEST(Simplify, HashLargeExpression) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  VarHandle i("i", kInt);
  auto memcpy_stmt = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          {i},
          CompareSelect::make(
              Load::make(a, {i}),
              Load::make(b, {i}),
              CompareSelectOperation::kEQ)));

  BufHandle d("D", {1}, kInt);
  BufHandle e("E", {1}, kInt);
  auto store_ramp_stmt = Store::make(
      e,
      {Ramp::make(0, 1, 4)},
      Load::make(d, {Ramp::make(0, 1, 4)}, Broadcast::make(IntImm::make(1), 4)),
      Broadcast::make(Cast::make(kInt, DoubleImm::make(1)), 4));

  auto if_stmt = Cond::make(
      CompareSelect::make(
          Load::make(a, {i}), Load::make(b, {i}), CompareSelectOperation::kGE),
      memcpy_stmt,
      store_ramp_stmt);

  HashProvider hasher;
  auto hash_r = hasher.hash(if_stmt);
  // We should not have to do any more work.
  ASSERT_TRUE(hasher.cachedHash(memcpy_stmt));
  auto hash_t = hasher.hash(memcpy_stmt);
  ASSERT_TRUE(hasher.cachedHash(store_ramp_stmt));
  auto hash_f = hasher.hash(store_ramp_stmt);

  // Root not equal to either branch, and branches not equal.
  ASSERT_NE(hash_r, hash_t);
  ASSERT_NE(hash_r, hash_f);
  ASSERT_NE(hash_t, hash_f);
}

TEST(Simplify, HashForLoopOptions) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  VarHandle i("i", kInt);
  auto for_stmt = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          {i},
          CompareSelect::make(
              Load::make(a, {i}),
              Load::make(b, {i}),
              CompareSelectOperation::kEQ)));

  HashProvider hasher;
  auto hash_before = hasher.hash(for_stmt);
  hasher.clearCache();

  for_stmt->set_gpu_block_index(LoopOptions::IDX_X);
  auto hash_block_idx = hasher.hash(for_stmt);
  hasher.clearCache();

  ASSERT_NE(hash_before, hash_block_idx);

  for_stmt->set_gpu_block_index(LoopOptions::IDX_UNSET);
  auto hash_reset = hasher.hash(for_stmt);
  hasher.clearCache();

  ASSERT_EQ(hash_before, hash_reset);
  for_stmt->set_gpu_thread_index(LoopOptions::IDX_X);
  auto hash_thread_idx = hasher.hash(for_stmt);

  ASSERT_NE(hash_before, hash_thread_idx);
  ASSERT_NE(hash_block_idx, hash_thread_idx);
}

/// (2 + x) + 4 => x + 6
TEST(Simplify, SimplifyAdd) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  VarHandle n_1("n_1", kInt);
  ExprHandle body = (ExprHandle(2) + x) + ExprHandle(4);

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Add* root = simplified.AsNode<Add>();
  ASSERT_NE(root, nullptr);
  const Var* lhs = dynamic_cast<const Var*>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->name_hint(), "x");
  const IntImm* rhs = dynamic_cast<const IntImm*>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->value(), 6.f);
}

/// (2 - x) - 4 => -2 - x
TEST(Simplify, SimplifySub) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body = (ExprHandle(2) - x) - ExprHandle(4);

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Sub* root = simplified.AsNode<Sub>();
  ASSERT_NE(root, nullptr);
  const IntImm* lhs = dynamic_cast<const IntImm*>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->value(), -2.f);
  const Var* rhs = dynamic_cast<const Var*>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->name_hint(), "x");
}

/// 2 * (1 - x) - 4 => 2 * (-3 - x)
TEST(Simplify, SimplifyMultiLayer) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body = ExprHandle(2) * ((ExprHandle(1) - x) - ExprHandle(4));
  ExprHandle simplified = IRSimplifier::simplify(body);
  IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
  IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
  IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
  IS_IMM_WITH_VAL(Int, sub->lhs(), -3);
  IS_VAR_WITH_NAME(sub->rhs(), "x");
}

/// 2 * (3 * x) - (x * 4) => 2 * x
TEST(Simplify, SimplifyMultiTerm) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body =
      (ExprHandle(2) * ((ExprHandle(3) * x)) - (x * ExprHandle(4)));

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Mul* root = simplified.AsNode<Mul>();
  ASSERT_NE(root, nullptr);
  const IntImm* lhs = dynamic_cast<const IntImm*>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->value(), 2);
  const Var* rhs = dynamic_cast<const Var*>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->name_hint(), "x");
}

/// 2 * (3 * (long)x) - (x * 4) => 2 * x
TEST(Simplify, SimplifyCasts) {
  KernelScope kernel_scope;
  VarHandle x("x", kLong);
  ExprHandle body =
      (ExprHandle(2) * ((ExprHandle(3) * x)) - (x * ExprHandle(4)));

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Mul* root = simplified.AsNode<Mul>();
  ASSERT_NE(root, nullptr);
  const LongImm* lhs = dynamic_cast<const LongImm*>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  ASSERT_EQ(lhs->value(), 2);
  const Var* rhs = dynamic_cast<const Var*>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  ASSERT_EQ(rhs->name_hint(), "x");
}

/// (x + 0) * 1 => x
TEST(Simplify, SimplifyEliminatesNoOps) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body = (x + ExprHandle(0)) * 1;

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Var* root = simplified.AsNode<Var>();
  ASSERT_NE(root, nullptr);
  ASSERT_EQ(root->name_hint(), "x");
}

/// Cannot simplify this.
TEST(Simplify, SimplifyMultiVar) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = x * 24 + y * 34;

  ExprHandle simplified = IRSimplifier::simplify(body);

  const Add* root = simplified.AsNode<Add>();
  ASSERT_NE(root, nullptr);
  const Mul* lhs = dynamic_cast<const Mul*>(root->lhs());
  ASSERT_NE(lhs, nullptr);
  const Var* varX = dynamic_cast<const Var*>(lhs->rhs());
  ASSERT_NE(varX, nullptr);
  ASSERT_EQ(varX->name_hint(), "y");
  const Mul* rhs = dynamic_cast<const Mul*>(root->rhs());
  ASSERT_NE(rhs, nullptr);
  const Var* varY = dynamic_cast<const Var*>(rhs->rhs());
  ASSERT_NE(varY, nullptr);
  ASSERT_EQ(varY->name_hint(), "x");
}

// x + 2 + y => x + y + 2
TEST(Simplify, DISABLED_SimplifyReorderings) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = x + 2 + y;
  ExprHandle simplified = IRSimplifier::simplify(body);

  const Add* root = simplified.AsNode<Add>();
  ASSERT_NE(root, nullptr);

  IS_NODE_WITH_NAME(Add, root->lhs(), rhs);
  IS_VAR_WITH_NAME(rhs->lhs(), "x");
  IS_VAR_WITH_NAME(rhs->rhs(), "y");
  IS_IMM_WITH_VAL(Int, root->rhs(), 2);
}

/// y + x * 0 => y
TEST(Simplify, SimplifyEliminatesVar) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = y + x * ExprHandle(0);

  ExprHandle simplified = IRSimplifier::simplify(body);
  IS_VAR_WITH_NAME(simplified.node(), "y");
}

TEST(Simplify, SimplifyAdds) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // (x + y) + (x + y) => 2 * (x + y)
    ExprHandle body = (x + y) + (x + y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), root);
    IS_IMM_WITH_VAL(Int, root->lhs(), 2);
    IS_NODE_WITH_NAME(Add, root->rhs(), add);
    IS_VAR_WITH_NAME(add->lhs(), "y");
    IS_VAR_WITH_NAME(add->rhs(), "x");
  }

  {
    // (x * y) + (x * y) => 2 * (x * y)
    ExprHandle body = (x * y) + (x * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), root);
    IS_IMM_WITH_VAL(Int, root->lhs(), 2);
    IS_NODE_WITH_NAME(Mul, root->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // (x - y) + (x - y) => 2 * (x - y)
    ExprHandle body = (x - y) + (x - y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    IS_NODE_WITH_NAME(Sub, mul->rhs(), rhs);
    IS_VAR_WITH_NAME(rhs->lhs(), "x");
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // (x + x + x + x) => 4 * x
    ExprHandle body = (x + x + x + x);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), root);
    IS_IMM_WITH_VAL(Int, root->lhs(), 4);
    IS_VAR_WITH_NAME(root->rhs(), "x");
  }

  {
    // (x + 0) => x.
    ExprHandle body = x + 0;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // (x + 0.f) => float(x).
    ExprHandle body = x + 0.f;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }
}

TEST(Simplify, SimplifyMuls) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // (x + y) * (x + y) => (x + y) * (x + y)
    // We don't attempt to simplify mulitplication of polynomials since the
    // result is only very rarely more efficient.
    ExprHandle body = (x + y) * (x + y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_NODE_WITH_NAME(Add, mul->lhs(), lhs);
    IS_VAR_WITH_NAME(lhs->lhs(), "y");
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_NODE_WITH_NAME(Add, mul->rhs(), rhs);
    IS_VAR_WITH_NAME(rhs->lhs(), "y");
    IS_VAR_WITH_NAME(rhs->rhs(), "x");
  }

  {
    // x * y * x * y => x * x * y * y
    // These get reordered only.
    ExprHandle body = x * y * x * y;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul1);
    IS_NODE_WITH_NAME(Mul, mul1->lhs(), mul2);
    IS_NODE_WITH_NAME(Mul, mul2->lhs(), mul3);
    IS_VAR_WITH_NAME(mul1->rhs(), "y");
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
    IS_VAR_WITH_NAME(mul3->lhs(), "x");
    IS_VAR_WITH_NAME(mul3->rhs(), "x");
  }

  {
    // 1 * (x * 1) => x
    // Ones cancel cleanly.
    ExprHandle body = ExprHandle(1) * (x * ExprHandle(1));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // 1.f * (x * 1.f) => x
    // Even float ones cancel cleanly, but carry their type.
    ExprHandle body = ExprHandle(1.f) * (x * ExprHandle(1.f));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }

  {
    // 1 * (x * 1.f) => x
    // One float is enough to cast the expr.
    ExprHandle body = ExprHandle(1) * (x * ExprHandle(1.f));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }

  {
    // 1 * (x * 0) => 0
    // Zeroes are eliminated.
    ExprHandle body = ExprHandle(1) * (x * ExprHandle(0));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // 1 * (x * 0) => 0
    // But not for Float since nan * 0 = nan.
    ExprHandle body = ExprHandle(1.f) * (x * ExprHandle(0.f));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_NODE_WITH_NAME(Cast, mul->lhs(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    IS_VAR_WITH_NAME(cast->src_value(), "x");
    IS_IMM_WITH_VAL(Float, mul->rhs(), 0.0);
  }

  {
    // (x - y) * (x - y) => (x - y) * (x - y)
    // As with Add we don't attempt simplification of this.
    ExprHandle body = (x - y) * (x - y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_NODE_WITH_NAME(Sub, mul->lhs(), lhs);
    IS_VAR_WITH_NAME(lhs->lhs(), "x");
    IS_VAR_WITH_NAME(lhs->rhs(), "y");
    IS_NODE_WITH_NAME(Sub, mul->rhs(), rhs);
    IS_VAR_WITH_NAME(rhs->lhs(), "x");
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // (x + y) * (x - y) => (x - y) * (x - y)
    // Don't simplify with different ops on each side.
    ExprHandle body = (x + y) * (x - y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_NODE_WITH_NAME(Add, mul->lhs(), lhs);
    IS_VAR_WITH_NAME(lhs->lhs(), "y");
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_NODE_WITH_NAME(Sub, mul->rhs(), rhs);
    IS_VAR_WITH_NAME(rhs->lhs(), "x");
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }
}

// Sub an expr from itself will result in zero.
TEST(Simplify, SimplifySubs) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // (x + y) - (x + y) => 0
    ExprHandle body = (x + y) - (x + y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // (x * y) - (x * y) => 0
    ExprHandle body = (x * y) - (x * y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // (x - y) - (x - y) => 0
    ExprHandle body = (x - y) - (x - y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // (x + y) - 2 * (x + y) => -1 * x - y
    ExprHandle body = (x + y) - ExprHandle(2) * (x + y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), -1);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
    IS_VAR_WITH_NAME(sub->rhs(), "y");
  }

  {
    // (x + y) - y => x
    ExprHandle body = (x + y) - y;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // (x - 0) => x.
    ExprHandle body = x - 0;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // (x - 0.f) => x.
    // Simple enough to cancel in float.
    ExprHandle body = x - ExprHandle(0.f);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }

  {
    // (x - (float)(y - y)) => x.
    ExprHandle body = x - Cast::make(kFloat, y - y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cast, simplified.node(), cast);
    ASSERT_EQ(cast->dtype().scalar_type(), ScalarType::Float);
    IS_VAR_WITH_NAME(cast->src_value(), "x");
  }

  {
    // (x - y) - y => x - 2 * y
    ExprHandle body = (x - y) - y;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_VAR_WITH_NAME(sub->lhs(), "x");
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // 2 * x - x => x
    ExprHandle body = (ExprHandle(2) * x) - x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // x - 2 * x = -1 * x
    // We don't have a unary negate, but this could be 0 -x I guess?
    ExprHandle body = x - (ExprHandle(2) * x);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);

    IS_IMM_WITH_VAL(Int, mul->lhs(), -1);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // (x + y + 5) * (x - x) => 0
    // Cancelling out one side of Mul cancels both.
    ExprHandle body = (x + y + 5) * (x - x);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // Cancel out opaque modulus.
    ExprHandle body = (x % y + 2) - (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 2);
  }

  {
    // Cancel out opaque modulus with a bit more going on.
    ExprHandle body = (x % y + (x * 2 - x - y * 0) - x + 2) - (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 2);
  }

  {
    // Sub where result is negative.
    ExprHandle body = x - (x + 1);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), -1);
  }

  {
    // Sub where result is positive due to negative scalar on RHS.
    ExprHandle body = x - (x - 1);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 1);
  }

  {
    // Term - Polynomial sub where RHS must be negated.
    ExprHandle body = (x * 2) - (x * 2 + 1);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), -1);
  }

  {
    // Term - Polynomial sub where the result is a Term.
    ExprHandle body = (y * x * 2) - (x * y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);

    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // Term - Polynomial sub where the result is a Polynomial.
    ExprHandle body = (x * 2) - (x + 1);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);

    IS_VAR_WITH_NAME(sub->lhs(), "x");
    IS_IMM_WITH_VAL(Int, sub->rhs(), 1);
  }
}

TEST(Simplify, SimplifyDiv) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);

  {
    ExprHandle body = ExprHandle(0) / x;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    ExprHandle body = x / 1;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_VAR_WITH_NAME(simplified.node(), "x");
  }
}

TEST(Simplify, SimplifyMod) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  {
    // Constant folding works.
    ExprHandle body = ExprHandle(10) % 8;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 2);
  }

  {
    // x % x => 0
    ExprHandle body = x % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // 0 % x => 0
    ExprHandle body = ExprHandle(0) % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // x % 1 => 0
    ExprHandle body = x % 1;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // Doesn't change unknown mods.
    // x % y => x % y
    ExprHandle body = x % y;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_VAR_WITH_NAME(mod->rhs(), "y");
  }

  {
    // don't touch if RHS is unknown.
    // 4 % x => 4 % x
    ExprHandle body = ExprHandle(4) % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_IMM_WITH_VAL(Int, mod->lhs(), 4);
    IS_VAR_WITH_NAME(mod->rhs(), "x");
  }

  {
    // don't touch if LHS is unknown.
    // x % 4 => x % 4
    ExprHandle body = x % 4;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 4);
  }

  {
    // if LHS is a multiple of RHS, mod is zero.
    // 2 * x % x => 0
    ExprHandle body = (x * 2) % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // true even if the multiple is not constant.
    // x * y % x => 0
    ExprHandle body = (x * y) % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // true with multiple unknown values in LHS.
    // x * y * z % x => 0
    ExprHandle body = (x * y * z) % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // true if the denom is compound.
    // x * y * z % y * z => 0
    ExprHandle body = (x * y * z) % (y * z);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // Sanity check true with scalars that are multiples.
    // 12 * x % 4 => 0
    ExprHandle body = (x * 12) % 4;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }

  {
    // Sanity check not true if the smaller scalar is on LHS.
    // 4 * x % 12 => 4 * x % 12
    ExprHandle body = (x * 4) % 12;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_NODE_WITH_NAME(Mul, mod->lhs(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 4);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 12);
  }

  {
    // Both scalar and symbolic in multiple.
    // (6 * x * y) % (3 * x * y) => 0
    ExprHandle body = (ExprHandle(6) * x * y) % (x * y * 3);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_IMM_WITH_VAL(Int, simplified.node(), 0);
  }
}

// Test that mixing ops together simplifies as expected.
TEST(Simplify, SimplifyMultiOp) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // (x * y) + (x - y) => (x + x * y) - y
    ExprHandle body = (x * y) + (x - y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Add, sub->lhs(), add);
    IS_VAR_WITH_NAME(add->lhs(), "x");
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
    IS_VAR_WITH_NAME(sub->rhs(), "y");
  }

  {
    // (x + y) - (x * y) => x + y - (x * y)
    ExprHandle body = (x + y) - (x * y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Add, sub->lhs(), add);
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul);
    IS_VAR_WITH_NAME(add->lhs(), "y");
    IS_VAR_WITH_NAME(add->rhs(), "x");
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // (x - y) - (x + y) => -2 * y
    ExprHandle body = (x - y) - (x + y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), -2);
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // (x - 0) + (x * 1) - (x + 0) => x
    ExprHandle body = (x - 0) + (x * 1) - (x + 0);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // (x - 0.f) + (x * 1.f) - (x + 0.f) => float(x) + float(x) - float(x)
    // Even in Float simple terms cancel out, but the variable ones cannot.
    ExprHandle body =
        (x - ExprHandle(0.f)) + (x * ExprHandle(1.f)) - (x + ExprHandle(0.f));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Add, sub->lhs(), add);
    IS_NODE_WITH_NAME(Cast, add->lhs(), cast1);
    IS_VAR_WITH_NAME(cast1->src_value(), "x");
    IS_NODE_WITH_NAME(Cast, add->rhs(), cast2);
    IS_VAR_WITH_NAME(cast2->src_value(), "x");
    IS_NODE_WITH_NAME(Cast, sub->rhs(), cast3);
    IS_VAR_WITH_NAME(cast3->src_value(), "x");
  }
}

// Test that chaining many ops together works as expected.
TEST(Simplify, SimplifyManyOps) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // x + y + x + x + y + y + x + y + x = 5 * x + 4 * y
    ExprHandle body = x + y + x + x + y + y + x + y + x;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Add, simplified.node(), add);

    IS_NODE_WITH_NAME(Mul, add->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 5);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");

    IS_NODE_WITH_NAME(Mul, add->rhs(), rhs);
    IS_IMM_WITH_VAL(Int, rhs->lhs(), 4);
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // x - y + x + x - y - y + x - y + x = 5 * x - 4 * y
    ExprHandle body = x - y + x + x - y - y + x - y + x;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), add);

    IS_NODE_WITH_NAME(Mul, add->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 5);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");

    IS_NODE_WITH_NAME(Mul, add->rhs(), rhs);
    IS_IMM_WITH_VAL(Int, rhs->lhs(), 4);
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // x + y + x - x - y - y + x + y + x = 3 * x
    ExprHandle body = x + y + x - x - y - y + x + y + x;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 3);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }
}

TEST(Simplify, SimplifyFactorization) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // (2 * x) + (2 * y) => 2 * (x + y)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(2) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    IS_NODE_WITH_NAME(Add, mul->rhs(), add);
    IS_VAR_WITH_NAME(add->lhs(), "y");
    IS_VAR_WITH_NAME(add->rhs(), "x");
  }

  {
    // Factorization when scalars have common divider.
    // (2 * x) + (4 * y) => 2 * (2 * y + x)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(4) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    IS_NODE_WITH_NAME(Add, mul->rhs(), add);
    IS_VAR_WITH_NAME(add->lhs(), "x");
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul2);
    IS_IMM_WITH_VAL(Int, mul2->lhs(), 2);
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // Factorization attempt without a common divider.
    // (2 * x) + (5 * y) =>  (5 * y) + (2 * x)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(5) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Add, simplified.node(), add);

    IS_NODE_WITH_NAME(Mul, add->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 5);
    IS_VAR_WITH_NAME(lhs->rhs(), "y");

    IS_NODE_WITH_NAME(Mul, add->rhs(), rhs);
    IS_IMM_WITH_VAL(Int, rhs->lhs(), 2);
    IS_VAR_WITH_NAME(rhs->rhs(), "x");
  }

  {
    // Factorization after merging.
    // (2 * x) + (4 * y) + (8 * x + 6 * y) => 10 * (x + y)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(4) * y) +
        (ExprHandle(8) * x + ExprHandle(6) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 10);

    IS_NODE_WITH_NAME(Add, mul->rhs(), add);
    IS_VAR_WITH_NAME(add->lhs(), "y");
    IS_VAR_WITH_NAME(add->rhs(), "x");
  }

  {
    // Factorization with common divider but different signs.
    // (2 * x) + (-4 * y) => 2 * (x - 2 * y)
    ExprHandle body = (ExprHandle(2) * x + ExprHandle(-4) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    IS_VAR_WITH_NAME(sub->lhs(), "x");
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul2);
    IS_IMM_WITH_VAL(Int, mul2->lhs(), 2);
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // Factorization with all negative numbers.
    // (-2 * x) + (-4 * y) => 2 * (-1 * x - 2 * y)
    ExprHandle body = ExprHandle(-2) * x + ExprHandle(-4) * y;
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);

    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), mul2);
    IS_IMM_WITH_VAL(Int, mul2->lhs(), -1);
    IS_VAR_WITH_NAME(mul2->rhs(), "x");
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul3);
    IS_IMM_WITH_VAL(Int, mul3->lhs(), 2);
    IS_VAR_WITH_NAME(mul3->rhs(), "y");
  }

  {
    // The following test ensures that there in no infinite recursion during
    // factorization when negative numbers are involved.
    VarHandle a("a", kInt);
    VarHandle b("b", kInt);
    VarHandle c("c", kInt);
    VarHandle d("d", kInt);
    VarHandle e("e", kInt);
    VarHandle f("f", kInt);
    VarHandle g("g", kInt);
    VarHandle h("h", kInt);

    ExprHandle body = ExprHandle(0) + (ExprHandle(1024) * a) +
        (ExprHandle(-1) * b) + (ExprHandle(-1) * c) + (ExprHandle(1) * d) +
        (ExprHandle(1) * e) + (ExprHandle(32) * f) + (ExprHandle(-1024) * g) +
        (ExprHandle(-32) * h);
    ExprHandle simplified = IRSimplifier::simplify(body);

    // We only check for the top level nodes here, since the main purpose
    // here is ensure that this simplification completes.
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 1024);
    IS_VAR_WITH_NAME(mul->rhs(), "g");
  }
}

// (4 * x + y + z * 2) + (4 * x + y + z * 4) => 2 * (y + 3 * z + 4 * x)
TEST(Simplify, SimplifyFactorizeUneven) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);
  ExprHandle body =
      (ExprHandle(4) * x + y + z * 2) + (ExprHandle(4) * x + y + z * 4);
  ExprHandle simplified = IRSimplifier::simplify(body);

  IS_NODE_WITH_NAME(Mul, simplified.node(), root);
  IS_IMM_WITH_VAL(Int, root->lhs(), 2);
  IS_NODE_WITH_NAME(Add, root->rhs(), add1);
  IS_NODE_WITH_NAME(Add, add1->lhs(), add2);

  IS_VAR_WITH_NAME(add2->lhs(), "y");
  IS_NODE_WITH_NAME(Mul, add2->rhs(), zmul);
  IS_NODE_WITH_NAME(Mul, add1->rhs(), xmul);

  IS_IMM_WITH_VAL(Int, xmul->lhs(), 4);
  IS_VAR_WITH_NAME(xmul->rhs(), "x");

  IS_IMM_WITH_VAL(Int, zmul->lhs(), 3);
  IS_VAR_WITH_NAME(zmul->rhs(), "z");
}

// (x * y) + (2 * x) * (x + y) => 3 * (x * y) + 2 * (x * x)
// This is kind of a placeholder test for variable factorization.
TEST(Simplify, SimplifyDeeperTerms) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = (x * y) + (ExprHandle(2) * x) * (x + y);
  ExprHandle simplified = IRSimplifier::simplify(body);

  IS_NODE_WITH_NAME(Add, simplified.node(), add);

  IS_NODE_WITH_NAME(Mul, add->lhs(), lhs);
  IS_IMM_WITH_VAL(Int, lhs->lhs(), 3);
  IS_NODE_WITH_NAME(Mul, lhs->rhs(), xyTerm);
  IS_VAR_WITH_NAME(xyTerm->lhs(), "x");
  IS_VAR_WITH_NAME(xyTerm->rhs(), "y");

  IS_NODE_WITH_NAME(Mul, add->rhs(), rhs);
  IS_IMM_WITH_VAL(Int, rhs->lhs(), 2);
  IS_NODE_WITH_NAME(Mul, rhs->rhs(), xxTerm);
  IS_VAR_WITH_NAME(xxTerm->rhs(), "x");
  IS_VAR_WITH_NAME(xxTerm->rhs(), "x");
}

// Tests the difference between two less trivial expressions.
// (m * (1 * n_1) + (n  + 1)) - (m *  (1 * n_1) + n) => 1
TEST(Simplify, SimplifyDeeperDifference) {
  KernelScope kernel_scope;
  VarHandle n("n", kInt);
  VarHandle n_1("n_1", kInt);
  VarHandle m("m", kInt);
  ExprHandle body =
      (m * (ExprHandle(1) * n_1) + (n + 1)) - (m * (ExprHandle(1) * n_1) + n);
  ExprHandle simplified = IRSimplifier::simplify(body);

  IS_IMM_WITH_VAL(Int, simplified.node(), 1);
}

// Test constant folding into the difference between expressions.
// 2 + char((m * (1 * n_1) + (n  + 1)) - (m *  (1 * n_1) + n)) => 3
TEST(Simplify, SimplifyFoldComplexDifference) {
  KernelScope kernel_scope;
  VarHandle n("n", kInt);
  VarHandle n_1("n_1", kInt);
  VarHandle m("m", kInt);
  ExprHandle body =
      (IntImm::make(2) +
       (Cast::make(
           kChar,
           (m * (ExprHandle(1) * n_1) + (n + 1)) -
               (m * (ExprHandle(1) * n_1) + n))));
  ExprHandle simplified = IRSimplifier::simplify(body);
  IS_IMM_WITH_VAL(Int, simplified.node(), 3);
}

TEST(Simplify, SimplifyIfComponents) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = IfThenElse::make(
      ((ExprHandle(5) - ExprHandle(4)) * x) > y,
      ExprHandle(2) * x - x,
      ExprHandle(2) * y - y);

  ExprHandle simplified = IRSimplifier::simplify(body);

  IS_NODE_WITH_NAME(IfThenElse, simplified.node(), ifexpr);

  IS_NODE_WITH_NAME(CompareSelect, ifexpr->condition(), cmp);
  ASSERT_EQ(cmp->compare_select_op(), kGT);
  IS_VAR_WITH_NAME(cmp->lhs(), "x");
  IS_VAR_WITH_NAME(cmp->rhs(), "y");

  IS_VAR_WITH_NAME(ifexpr->true_value(), "x");
  IS_VAR_WITH_NAME(ifexpr->false_value(), "y");
}

TEST(Simplify, SimplifyOpaqueTerms) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  {
    // 2 * x/y * x - x/y * y => y * x/y
    ExprHandle body = ((ExprHandle(2)) * (x / y) * y) - ((x / y) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "y");
    IS_NODE_WITH_NAME(Div, mul->rhs(), div);
    IS_VAR_WITH_NAME(div->lhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
  }

  {
    // x%y - (x%y - 1) => 1
    ExprHandle body = (x % y) - ((x % y) - 1);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_IMM_WITH_VAL(Int, simplified.node(), 1);
  }
}

TEST(Simplify, SimplifySymbolicMinMax) {
  KernelScope kernel_scope;

  {
    // Minimum with constant difference between terms.
    VarHandle x("x", kInt);
    ExprHandle body = Min::make(x + 3, x + 7, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->lhs(), "x");
    IS_IMM_WITH_VAL(Int, add->rhs(), 3);
  }

  {
    // Maximum with constant difference between terms.
    VarHandle x("x", kInt);
    ExprHandle body = Max::make(x + 3, x + 7, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->lhs(), "x");
    IS_IMM_WITH_VAL(Int, add->rhs(), 7);
  }

  {
    // Can't simplify multiples because of signedness of variable component.
    // TODO: maybe we could for unsigned types?
    VarHandle x("x", kInt);
    ExprHandle body = Max::make(x * 3, x * 7, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE(Max, simplified.node());
  }
}

TEST(Simplify, SimplifyNestedMax) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  {
    // Max(x + y, x + y) => x + y
    ExprHandle body = Max::make(x + y, x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_VARS(Add, simplified.node(), add, "y", "x");
  }

  {
    // Max(x + y, Max(x + y, z)) => Max(y + x, z)
    ExprHandle body = Max::make(x + y, Max::make(x + y, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_BINOP_W_VARS(Add, max->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // Max(x + y, Max(z, x + y)) => Max(y + x, z)
    ExprHandle body = Max::make(x + y, Max::make(z, x + y, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_BINOP_W_VARS(Add, max->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // Max(Max(x + y, z), x + y) => Max(y + x, z)
    ExprHandle body = Max::make(Max::make(x + y, z, true), x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_BINOP_W_VARS(Add, max->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // Max(Max(z, x + y), x + y) => Max(y + x, z)
    ExprHandle body = Max::make(Max::make(z, x + y, true), x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_BINOP_W_VARS(Add, max->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(max->rhs(), "z");
  }

  {
    // Max(Max(x, y), x) => Max(Max(x, y), x)
    // Nested Max ops with different propagate_nans should not be simplified.
    ExprHandle body = Max::make(Max::make(x, y, true), x, false);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_BINOP_W_VARS(Max, max->lhs(), max1, "x", "y");
    ASSERT_TRUE(max1->propagate_nans());
    IS_VAR_WITH_NAME(max->rhs(), "x");
    ASSERT_FALSE(max->propagate_nans());
  }

  {
    // Max(Min(x, y), Min(x, z)) => Min(x, Max(y, z))
    ExprHandle body =
        Max::make(Min::make(x, y, true), Min::make(x, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_VAR_WITH_NAME(min->lhs(), "x");
    IS_BINOP_W_VARS(Max, min->rhs(), max, "y", "z");
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(Min(x, y), Min(z, x)) => Min(x, Max(y, z))
    ExprHandle body =
        Max::make(Min::make(x, y, true), Min::make(z, x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_VAR_WITH_NAME(min->lhs(), "x");
    IS_BINOP_W_VARS(Max, min->rhs(), max, "y", "z");
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(Min(y, x), Min(x, z)) => Min(x, Max(y, z))
    ExprHandle body =
        Max::make(Min::make(y, x, true), Min::make(x, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_VAR_WITH_NAME(min->lhs(), "x");
    IS_BINOP_W_VARS(Max, min->rhs(), max, "y", "z");
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(Min(y, x), Min(z, x)) => Min(x, Max(y, z))
    ExprHandle body =
        Max::make(Min::make(y, x, true), Min::make(z, x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_VAR_WITH_NAME(min->lhs(), "x");
    IS_BINOP_W_VARS(Max, min->rhs(), max, "y", "z");
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(Min(y, x), Min(z, x)) => Max(Min(x, z), Min(x, y))
    // When all the ops in the pattern do not have the same propagate_nans,
    // it should not be simplified.
    ExprHandle body =
        Max::make(Min::make(y, x, true), Min::make(z, x, false), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_BINOP_W_VARS(Min, max->lhs(), min1, "x", "z");
    ASSERT_FALSE(min1->propagate_nans());
    IS_BINOP_W_VARS(Min, max->rhs(), min2, "x", "y");
    ASSERT_TRUE(min2->propagate_nans());
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(5, Max(x, 8)) => Max(x, 8)
    ExprHandle body = Max::make(5, Max::make(x, 8, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(8, Max(x, 5)) => Max(x, 8)
    ExprHandle body = Max::make(8, Max::make(x, 5, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(Max(x, 8), 5) => Max(x, 8)
    ExprHandle body = Max::make(Max::make(x, 8, true), 5, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(Max(x, 5), 8) => Max(x, 8)
    ExprHandle body = Max::make(Max::make(x, 5, true), 8, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Max, simplified.node(), max, "x", 8);
    ASSERT_TRUE(max->propagate_nans());
  }

  {
    // Max(5, Max(x, Max(y, Max(z, 8)))) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        5, Max::make(x, Max::make(y, Max::make(z, 8, true), true), true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // Max(8, Max(Max(y, Max(z, 5)), x)) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        8, Max::make(Max::make(y, Max::make(z, 5, true), true), x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // Max(5, Max(Max(Max(z, 8), y), x)) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        5, Max::make(Max::make(Max::make(z, 8, true), y, true), x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // Max(Max(x, Max(y, Max(5, z))), 8) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        Max::make(x, Max::make(y, Max::make(5, z, true), true), true), 8, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // Max(Max(Max(y, Max(8, z)), x), 5) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        Max::make(Max::make(y, Max::make(z, 8, true), true), x, true), 5, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // Max(Max(Max(Max(5, z), y), x), 8) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        Max::make(Max::make(Max::make(z, 5, true), y, true), x, true), 8, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // Max(Max(Max(Max(z, 5), y), x), 8) => Max(Max(x, Max(Max(z, 5), y)), 8)
    // Do not simplify when all the Max ops do not have the same
    // propagate_nans.
    ExprHandle body = Max::make(
        Max::make(Max::make(Max::make(z, 5, true), y, false), x, true),
        8,
        false);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_VAR_WITH_NAME(max2->lhs(), "x");
    IS_NODE_WITH_NAME(Max, max2->rhs(), max3);
    IS_BINOP_W_CONST(Max, max3->lhs(), max4, "z", 5);
    ASSERT_TRUE(max4->propagate_nans());
    IS_VAR_WITH_NAME(max3->rhs(), "y");
    ASSERT_FALSE(max3->propagate_nans());
    ASSERT_TRUE(max2->propagate_nans());
    IS_IMM_WITH_VAL(Int, max1->rhs(), 8);
    ASSERT_FALSE(max1->propagate_nans());
  }

  {
    // Max(8, Max(Max(x, 5), Max(y, z))) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        8, Max::make(Max::make(x, 5, true), Max::make(y, z, true), true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }

  {
    // Max(Max(Max(x, 5), Max(y, z)), 8) => Max(Max(Max(x, 8), y), z)
    ExprHandle body = Max::make(
        Max::make(Max::make(x, 5, true), Max::make(y, z, true), true), 8, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max1);
    IS_NODE_WITH_NAME(Max, max1->lhs(), max2);
    IS_BINOP_W_CONST(Max, max2->lhs(), max3, "x", 8);
    ASSERT_TRUE(max3->propagate_nans());
    IS_VAR_WITH_NAME(max2->rhs(), "y");
    IS_VAR_WITH_NAME(max1->rhs(), "z");
  }
}

TEST(Simplify, SimplifyNestedMin) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  VarHandle z("z", kInt);

  {
    // Min(x + y, x + y) => x + y
    ExprHandle body = Min::make(x + y, x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_VARS(Add, simplified.node(), add, "y", "x");
  }

  {
    // Min(x + y, Min(x + y, z)) => Min(y + x, z)
    ExprHandle body = Min::make(x + y, Min::make(x + y, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(x + y, Min(z, x + y)) => Min(y + x, z)
    ExprHandle body = Min::make(x + y, Min::make(z, x + y, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(Min(x + y, z), x + y) => Min(y + x, z)
    ExprHandle body = Min::make(Min::make(x + y, z, true), x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(Min(z, x + y), x + y) => Min(y + x, z)
    ExprHandle body = Min::make(Min::make(z, x + y, true), x + y, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Add, min->lhs(), add, "y", "x");
    IS_VAR_WITH_NAME(min->rhs(), "z");
  }

  {
    // Min(Min(x, y), x) => Min(Min(x, y), x)
    // Nested Min ops with different propagate_nans should not be simplified.
    ExprHandle body = Min::make(Min::make(x, y, true), x, false);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_BINOP_W_VARS(Min, min1->lhs(), min2, "x", "y");
    ASSERT_TRUE(min2->propagate_nans());
    IS_VAR_WITH_NAME(min1->rhs(), "x");
    ASSERT_FALSE(min1->propagate_nans());
  }

  {
    // Min(Max(x, y), Max(x, z)) => Max(x, Min(y, z))
    ExprHandle body =
        Min::make(Max::make(x, y, true), Max::make(x, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_VAR_WITH_NAME(max->lhs(), "x");
    IS_BINOP_W_VARS(Min, max->rhs(), min, "y", "z");
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Max(x, y), Max(z, x)) => Max(x, Min(y, z))
    ExprHandle body =
        Min::make(Max::make(x, y, true), Max::make(z, x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_VAR_WITH_NAME(max->lhs(), "x");
    IS_BINOP_W_VARS(Min, max->rhs(), min, "y", "z");
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Max(y, x), Max(x, z)) => Max(x, Min(y, z))
    ExprHandle body =
        Min::make(Max::make(y, x, true), Max::make(x, z, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_VAR_WITH_NAME(max->lhs(), "x");
    IS_BINOP_W_VARS(Min, max->rhs(), min, "y", "z");
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Max(y, x), Max(z, x)) => Max(x, Min(y, z))
    ExprHandle body =
        Min::make(Max::make(y, x, true), Max::make(z, x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Max, simplified.node(), max);
    IS_VAR_WITH_NAME(max->lhs(), "x");
    IS_BINOP_W_VARS(Min, max->rhs(), min, "y", "z");
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Max(y, x), Max(z, x)) => Min(Max(x, z), Max(x, y))
    // When all the ops in the pattern do not have the same propagate_nans,
    // it should not be simplified.
    ExprHandle body =
        Min::make(Max::make(y, x, true), Max::make(z, x, false), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min);
    IS_BINOP_W_VARS(Max, min->lhs(), max1, "x", "z");
    ASSERT_FALSE(max1->propagate_nans());
    IS_BINOP_W_VARS(Max, min->rhs(), max2, "x", "y");
    ASSERT_TRUE(max2->propagate_nans());
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(5, Min(x, 8)) => Min(x, 8)
    ExprHandle body = Min::make(5, Min::make(x, 8, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(8, Min(x, 5)) => Min(x, 8)
    ExprHandle body = Min::make(8, Min::make(x, 5, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Min(x, 8), 5) => Min(x, 8)
    ExprHandle body = Min::make(Min::make(x, 8, true), 5, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(Min(x, 5), 8) => Min(x, 8)
    ExprHandle body = Min::make(Min::make(x, 5, true), 8, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_BINOP_W_CONST(Min, simplified.node(), min, "x", 5);
    ASSERT_TRUE(min->propagate_nans());
  }

  {
    // Min(5, Min(x, Min(y, Min(z, 8)))) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        5, Min::make(x, Min::make(y, Min::make(z, 8, true), true), true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(5, Min(Min(y, Min(z, 8)), x)) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        5, Min::make(Min::make(y, Min::make(z, 8, true), true), x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(5, Min(Min(Min(z, 8), y), x)) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        5, Min::make(Min::make(Min::make(z, 8, true), y, true), x, true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(Min(x, Min(y, Min(8, z))), 5) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        Min::make(x, Min::make(y, Min::make(8, z, true), true), true), 5, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(Min(Min(y, Min(8, z)), x), 5) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        Min::make(Min::make(y, Min::make(z, 8, true), true), x, true), 5, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(Min(Min(Min(8, z), y), x), 5) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        Min::make(Min::make(Min::make(z, 8, true), y, true), x, true), 5, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(Min(Min(Min(z, 5), y), x), 8) => Min(Min(x, Min(Min(z, 5), y)), 8)
    // Do not simplify when all the Min ops do not have the same
    // propagate_nans.
    ExprHandle body = Min::make(
        Min::make(Min::make(Min::make(z, 5, true), y, false), x, true),
        8,
        false);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_VAR_WITH_NAME(min2->lhs(), "x");
    IS_NODE_WITH_NAME(Min, min2->rhs(), min3);
    IS_BINOP_W_CONST(Min, min3->lhs(), min4, "z", 5);
    ASSERT_TRUE(min4->propagate_nans());
    IS_VAR_WITH_NAME(min3->rhs(), "y");
    ASSERT_FALSE(min3->propagate_nans());
    ASSERT_TRUE(min2->propagate_nans());
    IS_IMM_WITH_VAL(Int, min1->rhs(), 8);
    ASSERT_FALSE(min1->propagate_nans());
  }

  {
    // Min(8, Min(Min(x, 5), Min(y, z))) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        8, Min::make(Min::make(x, 5, true), Min::make(y, z, true), true), true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }

  {
    // Min(Min(Min(x, 5), Min(y, z)), 8) => Min(Min(Min(x, 5), y), z)
    ExprHandle body = Min::make(
        Min::make(Min::make(x, 5, true), Min::make(y, z, true), true), 8, true);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Min, simplified.node(), min1);
    IS_NODE_WITH_NAME(Min, min1->lhs(), min2);
    IS_BINOP_W_CONST(Min, min2->lhs(), min3, "x", 5);
    ASSERT_TRUE(min3->propagate_nans());
    IS_VAR_WITH_NAME(min2->rhs(), "y");
    IS_VAR_WITH_NAME(min1->rhs(), "z");
  }
}

TEST(Simplify, SimplifyWontReorderFloat) {
  KernelScope kernel_scope;

  {
    // 3 * (3 * x) - 3 * (3 * y) => 9 * (x - y)
    // This is an expression we can simplify.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);

    ExprHandle body = ExprHandle(3) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 9);
    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    IS_VAR_WITH_NAME(sub->lhs(), "x");
    IS_VAR_WITH_NAME(sub->rhs(), "y");
  }

  {
    // 3 * (3 * x) - 3 * (3 * y) => 3 * (3 * x) - 3 * (3 * y).
    // If the vars are floating point, ops are not associative and we can't
    // reorder.
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);

    ExprHandle body = ExprHandle(3) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), lhsMul);
    IS_IMM_WITH_VAL(Float, lhsMul->lhs(), 3);
    IS_NODE_WITH_NAME(Mul, lhsMul->rhs(), lhsVarMul);
    IS_IMM_WITH_VAL(Float, lhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(lhsVarMul->rhs(), "x");

    IS_NODE_WITH_NAME(Mul, sub->rhs(), rhsMul);
    IS_IMM_WITH_VAL(Float, rhsMul->lhs(), 3);
    IS_NODE_WITH_NAME(Mul, rhsMul->rhs(), rhsVarMul);
    IS_IMM_WITH_VAL(Float, rhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(rhsVarMul->rhs(), "y");
  }

  {
    // 3 * (3 * x) - 3 * (3 * y) => 3 * (3 * x) - (9 * y).
    // We will simplify subexprs if they dont reorder floating point ops.
    VarHandle x("x", kDouble);
    VarHandle y("y", kInt);

    ExprHandle body = ExprHandle(3) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), lhsMul);
    IS_IMM_WITH_VAL(Double, lhsMul->lhs(), 3);
    IS_NODE_WITH_NAME(Mul, lhsMul->rhs(), lhsVarMul);
    IS_IMM_WITH_VAL(Double, lhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(lhsVarMul->rhs(), "x");

    IS_NODE_WITH_NAME_AND_CAST(Mul, sub->rhs(), rhsMul, Double);
    IS_IMM_WITH_VAL(Int, rhsMul->lhs(), 9);
    IS_VAR_WITH_NAME(rhsMul->rhs(), "y");
  }

  {
    // Prevent reordering if FP propagated from dtypes.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);

    ExprHandle body = ExprHandle(3.f) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3.f) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mul, sub->lhs(), lhsMul);
    IS_IMM_WITH_VAL(Float, lhsMul->lhs(), 3);
    IS_NODE_WITH_NAME_AND_CAST(Mul, lhsMul->rhs(), lhsVarMul, Float);
    IS_IMM_WITH_VAL(Int, lhsVarMul->lhs(), 3);
    IS_VAR_WITH_NAME(lhsVarMul->rhs(), "x");

    IS_NODE_WITH_NAME(Mul, sub->rhs(), rhsMul);
    IS_IMM_WITH_VAL(Float, rhsMul->lhs(), 3);
    IS_NODE_WITH_NAME(Mul, rhsMul->rhs(), rhsVarMul);
    IS_IMM_WITH_VAL(Float, rhsVarMul->lhs(), 3);
    IS_NODE_WITH_NAME(Cast, rhsVarMul->rhs(), yCast);
    IS_VAR_WITH_NAME(yCast->src_value(), "y");
  }

  {
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    // x%y - (x%y - 1) => x%y - (x%y - 1).
    // We wont reorder opaque ops if they are FP.
    ExprHandle body = (x % y) - ((x % y) - 1);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_NODE_WITH_NAME(Mod, sub->lhs(), lhsMod);
    IS_VAR_WITH_NAME(lhsMod->lhs(), "x");
    IS_VAR_WITH_NAME(lhsMod->rhs(), "y");

    IS_NODE_WITH_NAME(Sub, sub->rhs(), rhsSub);
    IS_NODE_WITH_NAME(Mod, rhsSub->lhs(), rhsMod);
    IS_VAR_WITH_NAME(rhsMod->lhs(), "x");
    IS_VAR_WITH_NAME(rhsMod->rhs(), "y");
    IS_IMM_WITH_VAL(Float, rhsSub->rhs(), 1);
  }
}

TEST(Simplify, SimplifyRoundModPattern) {
  KernelScope kernel_scope;

  {
    // (x/y)*y + x%y => x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ((x / y) * y) + (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Reverse order.
    // x%y + (x/y)*y => x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (x % y) + ((x / y) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Non opaque denominator.
    // (x / (4+y)) * (4+y)) + (x % (y + 4)) => x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ((x / (ExprHandle(4) + y)) * (ExprHandle(4) + y)) +
        (x % (y + ExprHandle(4)));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Reverse order.
    // (x % (y + 4)) + (x / (4+y)) * (4+y)) => x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (x % (y + ExprHandle(4))) +
        ((x / (ExprHandle(4) + y)) * (ExprHandle(4) + y));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Opaque denominator.
    // (x / (2/y)) * (2/y)) + (x % (2/y)) => x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ((x / (ExprHandle(2) / y)) * (ExprHandle(2) / y)) +
        (x % (ExprHandle(2) / y));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Non opaque numerator
    // ((2*x)/y * y) + ((2*x) % y) => 2 * x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body =
        (((ExprHandle(2) * x) / y) * y) + ((ExprHandle(2) * x) % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Opaque numerator.
    // ((x/2) / y * y) + (x/2 % y) => x / 2.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body =
        (((x / ExprHandle(2)) / y) * y) + ((x / ExprHandle(2)) % y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_VAR_WITH_NAME(div->lhs(), "x");
    IS_IMM_WITH_VAL(Int, div->rhs(), 2);
  }

  {
    // Numerator and denominator.
    // ((2*x)/(2*y) * (2*y)) + ((2*x) % (2*y)) => 2 * x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body =
        (((ExprHandle(2) * x) / (ExprHandle(2) * y)) * (ExprHandle(2) * y)) +
        ((ExprHandle(2) * x) % (ExprHandle(2) * y));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Reverse order.
    // ((2*x) % (2*y)) + ((2*x)/(2*y) * (2*y)) => 2 * x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ((ExprHandle(2) * x) % (ExprHandle(2) * y)) +
        (((ExprHandle(2) * x) / (ExprHandle(2) * y)) * (ExprHandle(2) * y));
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Negated Subtraction of Round Mod.
    // (x/y) * y - (0 - x%y) => x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ((x / y) * y) - (ExprHandle(0) - (x % y));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // Other terms are preserved.
    // (x/y)*y + x%y + (y * x) => x + (y * x).
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ((x / y) * y) + (x % y) + (y * x);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->lhs(), "x");
    IS_NODE_WITH_NAME(Mul, add->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // Sanity checking we wont do the optimization on floats.
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    ExprHandle body = ((x / y) * y) + (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mul, add->lhs(), roundMul);
    IS_NODE_WITH_NAME(Div, roundMul->lhs(), roundDiv);
    IS_VAR_WITH_NAME(roundDiv->lhs(), "x");
    IS_VAR_WITH_NAME(roundDiv->rhs(), "y");
    IS_VAR_WITH_NAME(roundMul->rhs(), "y");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_VAR_WITH_NAME(mod->rhs(), "y");
  }

  {
    // Sanity check we wont do it if the mod term doesn't match.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);
    ExprHandle body = ((x / y) * y) + (x % z);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mul, add->lhs(), roundMul);
    IS_VAR_WITH_NAME(roundMul->lhs(), "y");
    IS_NODE_WITH_NAME(Div, roundMul->rhs(), roundDiv);
    IS_VAR_WITH_NAME(roundDiv->lhs(), "x");
    IS_VAR_WITH_NAME(roundDiv->rhs(), "y");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_VAR_WITH_NAME(mod->rhs(), "z");
  }

  {
    // Sanity check we wont do it if the div term doesn't match.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);
    ExprHandle body = (y * (x / z)) + (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mul, add->lhs(), roundMul);
    IS_VAR_WITH_NAME(roundMul->lhs(), "y");
    IS_NODE_WITH_NAME(Div, roundMul->rhs(), roundDiv);
    IS_VAR_WITH_NAME(roundDiv->lhs(), "x");
    IS_VAR_WITH_NAME(roundDiv->rhs(), "z");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_VAR_WITH_NAME(mod->rhs(), "y");
  }

  {
    // Sanity check we wont do it if the mul term doesn't match.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);
    ExprHandle body = ((x / y) * z) + (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mul, add->lhs(), roundMul);
    IS_VAR_WITH_NAME(roundMul->lhs(), "z");
    IS_NODE_WITH_NAME(Div, roundMul->rhs(), roundDiv);
    IS_VAR_WITH_NAME(roundDiv->lhs(), "x");
    IS_VAR_WITH_NAME(roundDiv->rhs(), "y");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_VAR_WITH_NAME(mod->rhs(), "y");
  }
}

TEST(Simplify, SimplifyRoundModPatternFactorization) {
  KernelScope kernel_scope;

  {
    // Full factorization.
    // 2 * (x/y * y) + 2 * (x%y) => 2 * x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ExprHandle(2) * ((x / y) * y) + ExprHandle(2) * (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Partial Factorization.
    // 32 * (x/8) + 4 * (x % 8) => 4 * x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = ExprHandle(32) * (x / 8) + ExprHandle(4) * (x % 8);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 4);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    // Factorization requiring constant folding.
    // 20 * (x  / (16 / 2)) * 2 + (11 % 6) * (x % (7+1)) => 5 * x.
    VarHandle x("x", kInt);
    ExprHandle body = ExprHandle(40) * (x / (ExprHandle(16) / 2)) +
        (ExprHandle(11) % 6) * (x % (ExprHandle(7) + 1));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 5);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    VarHandle x("x", kInt);
    ExprHandle body = (x / 5) * 10 + ExprHandle(2) * (x % 5);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }

  {
    VarHandle x("x", kInt);
    ExprHandle body = (x / 10) * 0 + x % 5;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "x");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 5);
  }
}

TEST(Simplify, SimplifyRoundModPatternMultivar) {
  KernelScope kernel_scope;

  {
    // Multivar.
    // (x/8) * 8 + (y/5)*5 + x%8 + y%5 => y + x.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (x / ExprHandle(8) * ExprHandle(8)) +
        (y / ExprHandle(5) * ExprHandle(5)) + (x % 8) + (y % 5);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->lhs(), "y");
    IS_VAR_WITH_NAME(add->rhs(), "x");
  }

  {
    // Find the right var.
    // (y/8) * 8  x%8 + y%8 + z%8 => z%8 + x%8 + y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);
    ExprHandle body =
        (y / ExprHandle(8) * ExprHandle(8)) + (x % 8) + (y % 8) + (z % 8);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Add, add->lhs(), add2);
    IS_NODE_WITH_NAME(Mod, add2->lhs(), xMod);
    IS_VAR_WITH_NAME(xMod->lhs(), "x");
    IS_IMM_WITH_VAL(Int, xMod->rhs(), 8);
    IS_VAR_WITH_NAME(add2->rhs(), "y");
    IS_NODE_WITH_NAME(Mod, add->rhs(), zMod);
    IS_VAR_WITH_NAME(zMod->lhs(), "z");
    IS_IMM_WITH_VAL(Int, zMod->rhs(), 8);
  }

  {
    // Compound.
    // (x + (z + 512 * y) % 16) + 16 * ((z + 512 * y) / 16)
    // => (z + 512 * y) + x
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle z("z", kInt);

    ExprHandle body = x + (z + ExprHandle(512) * y) % ExprHandle(16) +
        ExprHandle(16) * ((z + ExprHandle(512) * y) / ExprHandle(16));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->rhs(), "x");
    IS_NODE_WITH_NAME(Add, add->lhs(), add2);
    IS_VAR_WITH_NAME(add2->lhs(), "z");
    IS_NODE_WITH_NAME(Mul, add2->rhs(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 512);
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }
}

TEST(Simplify, SimplifyModRoundModPattern) {
  KernelScope kernel_scope;

  {
    // t/7 % 9 * 7 + t % 7 => t%63
    VarHandle t("t", kInt);
    ExprHandle body = (t / 7 % 9) * 7 + t % 7;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
  }

  {
    // 2*t/7 % 9 * 7 + 2*t % 7 => 2*t % 63
    VarHandle t("t", kInt);
    ExprHandle body = (ExprHandle(2) * t / 7 % 9) * 7 + ExprHandle(2) * t % 7;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_NODE_WITH_NAME(Mul, mod->lhs(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "t");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
  }

  {
    // t/x % y * x + t % x => t%(x*y)
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (t / x % y) * x + t % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // k*t/x % y * x + k*t % x => k*t%(x*y)
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    ExprHandle body = (k * t / x % y) * x + k * t % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_NODE_WITH_NAME(Mul, mod->lhs(), mul1);
    IS_VAR_WITH_NAME(mul1->lhs(), "t");
    IS_VAR_WITH_NAME(mul1->rhs(), "k");
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul2);
    IS_VAR_WITH_NAME(mul2->lhs(), "x");
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // t/k/x % y * x + t/k % x => t/k%(x*y)
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    ExprHandle body = (t / k / x % y) * x + t / k % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_NODE_WITH_NAME(Div, mod->lhs(), div);
    IS_VAR_WITH_NAME(div->lhs(), "t");
    IS_VAR_WITH_NAME(div->rhs(), "k");
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // Sanity checking we wont do the optimization on floats.
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    VarHandle z("z", kFloat);
    ExprHandle body = ((x / y % z) * y) + (x % y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mul, add->lhs(), mul);
    IS_NODE_WITH_NAME(Mod, mul->lhs(), mod);
    IS_NODE_WITH_NAME(Div, mod->lhs(), div);
    IS_VAR_WITH_NAME(div->lhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
    IS_VAR_WITH_NAME(mod->rhs(), "z");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod2);
    IS_VAR_WITH_NAME(mod2->lhs(), "x");
    IS_VAR_WITH_NAME(mod2->rhs(), "y");
  }
}

TEST(Simplify, SimplifyModRoundModPatternFactorization) {
  KernelScope kernel_scope;

  {
    // 2 * (t /7 % 9 * 7) + 2 * (t % 7) => 2 * (t % 63)
    VarHandle t("t", kInt);
    ExprHandle body =
        ExprHandle(2) * ((t / 7 % 9) * 7) + ExprHandle(2) * (t % 7);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_NODE_WITH_NAME(Mod, mul->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
  }

  {
    // t /7 % 9 * 14 + 2* (t % 7) => 2* (t % 63)
    VarHandle t("t", kInt);
    ExprHandle body = (t / 7 % 9) * 14 + ExprHandle(2) * (t % 7);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_NODE_WITH_NAME(Mod, mul->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
  }

  {
    // t/14 % 9 * 7 + t/2 % 7 => t/2 % 63
    VarHandle t("t", kInt);
    ExprHandle body = (t / 14 % 9) * 7 + t / 2 % 7;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_NODE_WITH_NAME(Div, mod->lhs(), div);
    IS_VAR_WITH_NAME(div->lhs(), "t");
    IS_IMM_WITH_VAL(Int, div->rhs(), 2);
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
  }

  {
    // t/(7*3) % 9 * 7*3 + t % (7*3) => t % 189
    VarHandle t("t", kInt);
    ExprHandle body = (t / (ExprHandle(7) * ExprHandle(3)) % 9) * 7 * 3 +
        t % (ExprHandle(7) * ExprHandle(3));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mod, simplified.node(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 189);
  }

  {
    // 2*(t/x % y * x) + 2*(t % x) => 2*(t%(x*y))
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body =
        ExprHandle(2) * ((t / x % y) * x) + ExprHandle(2) * (t % x);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_NODE_WITH_NAME(Mod, mul->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul2);
    IS_VAR_WITH_NAME(mul2->lhs(), "x");
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }
}

TEST(Simplify, SimplifyModRoundModPatternMultivar) {
  KernelScope kernel_scope;

  {
    // t/7 % 9 * 7 + t % 7 + t => t % 63 + t
    VarHandle t("t", kInt);
    ExprHandle body = (t / 7 % 9) * 7 + t % 7 + t;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod->rhs(), 63);
    IS_VAR_WITH_NAME(add->lhs(), "t");
  }

  {
    // t/7 % 9 * 7 + t/8 % 9 * 8 + t % 7 + t % 8  => t % 63 + t % 72
    VarHandle t("t", kInt);
    ExprHandle body = (t / 7 % 9) * 7 + (t / 8 % 9) * 8 + t % 7 + t % 8;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mod, add->lhs(), mod1);
    IS_VAR_WITH_NAME(mod1->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod1->rhs(), 63);
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod2);
    IS_VAR_WITH_NAME(mod2->lhs(), "t");
    IS_IMM_WITH_VAL(Int, mod2->rhs(), 72);
  }

  {
    // k + t/x % y * x + t % x => k + t%(x*y)
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    ExprHandle body = k + (t / x % y) * x + t % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_VAR_WITH_NAME(add->lhs(), "k");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
  }

  {
    // t/x % y * x + t % x + (t/k / x % y) * x + t/k % x
    // => t%(x*y) + t/k % (x*y)
    VarHandle t("t", kInt);
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    VarHandle k("k", kInt);
    ExprHandle body = (t / x % y) * x + t % x + (t / k / x % y) * x + t / k % x;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_NODE_WITH_NAME(Mod, add->lhs(), mod);
    IS_VAR_WITH_NAME(mod->lhs(), "t");
    IS_NODE_WITH_NAME(Mul, mod->rhs(), mul);
    IS_VAR_WITH_NAME(mul->lhs(), "x");
    IS_VAR_WITH_NAME(mul->rhs(), "y");
    IS_NODE_WITH_NAME(Mod, add->rhs(), mod2);
    IS_NODE_WITH_NAME(Div, mod2->lhs(), div);
    IS_VAR_WITH_NAME(div->lhs(), "t");
    IS_VAR_WITH_NAME(div->rhs(), "k");
    IS_NODE_WITH_NAME(Mul, mod2->rhs(), mul2);
    IS_VAR_WITH_NAME(mul2->lhs(), "x");
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }

  {
    // 3D: (7 * ((i0_flat / 7) % 9) + i0_flat % 7) + 63 * (i0_flat / 63)
    // => io_flat
    VarHandle t("io_flat", kInt);
    ExprHandle body =
        ExprHandle(7) * (t / 7 % 9) + t % 7 + ExprHandle(63) * (t / 63);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "io_flat");
  }

  { // 5D: i0_flat / (11 * 10 * 9 * 7)  * (7 * 9 * 10 * 11) +
    // (i0_flat / (10 * 9 * 7) % 11)  * 7 * 9 * 10 +
    // (i0_flat / (9 * 7) % 10) * 7 * 9 +
    // (i0_flat / 7 % 9)  * 7 +
    // i0_flat % 7 => io_flat
    VarHandle t("io_flat", kInt);
    ExprHandle body = (t / (ExprHandle(11) * 10 * 9 * 7)) * (7 * 9 * 10 * 11) +
        (t / (ExprHandle(10) * 9 * 7) % 11) * 7 * 9 * 10 +
        (t / (ExprHandle(9) * 7) % 10) * 7 * 9 + (t / 7 % 9) * 7 + t % 7;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "io_flat");
  }

  {
    // 3D: (m * ((i0_flat / m) % n) + i0_flat % m) + (m * n) *
    // (i0_flat / (m * n)) => io_flat
    VarHandle t("io_flat", kInt);
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    ExprHandle body = m * (t / m % n) + t % m + (m * n) * (t / (m * n));
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "io_flat");
  }

  { // 5D: i0_flat / (k * l * n * m)  * (m * n * l * k) +
    // (i0_flat / (l * n * m) % k)  * m * n * l +
    // (i0_flat / (n * m) % l) * m * n +
    // (i0_flat / m % n)  * m +
    // i0_flat % m => io_flat
    VarHandle t("io_flat", kInt);
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    VarHandle l("l", kInt);
    VarHandle k("k", kInt);
    ExprHandle body = (t / (k * l * n * m)) * (m * n * l * k) +
        (t / (l * n * m) % k) * m * n * l + (t / (n * m) % l) * m * n +
        (t / m % n) * m + t % m;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "io_flat");
  }
}

TEST(Simplify, SimplifyDivisionScalarFactorization) {
  KernelScope kernel_scope;

  {
    // Simple factorization of numerator and denominator.
    // 8x / 4y => 2x / y.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (x * 8) / (y * 4);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 2);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
  }

  {
    // Don't change anything if we can't factorize.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (x * 7) / (y * 4);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 7);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_NODE_WITH_NAME(Mul, div->rhs(), rhs);
    IS_IMM_WITH_VAL(Int, rhs->lhs(), 4);
    IS_VAR_WITH_NAME(rhs->rhs(), "y");
  }

  {
    // Don't reorder floats.
    VarHandle x("x", kFloat);
    VarHandle y("y", kFloat);
    ExprHandle body = (x * 8) / (y * 4);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_VAR_WITH_NAME(lhs->lhs(), "x");
    IS_IMM_WITH_VAL(Float, lhs->rhs(), 8.f);
    IS_NODE_WITH_NAME(Mul, div->rhs(), rhs);
    IS_VAR_WITH_NAME(rhs->lhs(), "y");
    IS_IMM_WITH_VAL(Float, rhs->rhs(), 4.f);
  }

  {
    // Sanity check we do nothing if there are only scalar parts.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (x * 1) / (y * 1);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_VAR_WITH_NAME(div->lhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
  }

  {
    // Can factorize amounts of variables.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = (x + x + x + x) / (y + y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Div, simplified.node(), div);
    IS_NODE_WITH_NAME(Mul, div->lhs(), lhs);
    IS_IMM_WITH_VAL(Int, lhs->lhs(), 2);
    IS_VAR_WITH_NAME(lhs->rhs(), "x");
    IS_VAR_WITH_NAME(div->rhs(), "y");
  }
}

TEST(Simplify, SimplifyConstantBranches) {
  KernelScope kernel_scope;

  {
    // If the condition is constant true then take the true_value.
    // 1 ? x : y => x
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle t(1);
    ExprHandle body = IfThenElse::make(t, x, y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // If the condition is constant false then take the false_value.
    // 0 ? x : y => y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle t(0);
    ExprHandle body = IfThenElse::make(t, x, y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "y");
  }

  {
    // condition is simplified before checking.
    // (x-x) ? x : y => y
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = IfThenElse::make(x - x, x, y);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "y");
  }

  {
    // If both branches are the same then don't do the condition.
    // y ? x : x => x
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = IfThenElse::make(y, x, x);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
  }

  {
    // If both branches simplify to the same thing it still works.
    // y ? (x + x) : (2 * x) => x
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);
    ExprHandle body = IfThenElse::make(y, x + x, ExprHandle(2) * x);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), 2);
    IS_VAR_WITH_NAME(mul->rhs(), "x");
  }
}

TEST(Simplify, SimplifyConstantCond) {
  KernelScope kernel_scope;

  {
    // If the condition is constant true then take the true_value.
    // 1 ? A[0] = 1 : B[0] = 1 => A[0] = 1
    BufHandle a("A", {1}, kInt);
    BufHandle b("B", {1}, kInt);
    ExprHandle condition(1);
    Stmt* true_val = Store::make(a, {0}, 1, 1);
    Stmt* false_val = Store::make(b, {0}, 1, 1);

    Cond* body = new Cond(condition.node(), true_val, false_val);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "A");
  }

  {
    // If the condition is constant false then take the false_value.
    // 0 ? A[0] = 1 : B[0] = 1 => B[0] = 1
    BufHandle a("A", {1}, kInt);
    BufHandle b("B", {1}, kInt);
    ExprHandle condition(0);
    Stmt* true_val = Store::make(a, {0}, 1, 1);
    Stmt* false_val = Store::make(b, {0}, 1, 1);

    Stmt* body = new Cond(condition.node(), true_val, false_val);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "B");
  }

  {
    // condition is simplified before checking.
    // (x-x) ? A[0] = 1 : B[0] = 1 => B[0] = 1
    VarHandle x("x", kInt);
    BufHandle a("A", {1}, kInt);
    BufHandle b("B", {1}, kInt);
    ExprHandle condition(x - x);
    Stmt* true_val = Store::make(a, {0}, 1, 1);
    Stmt* false_val = Store::make(b, {0}, 1, 1);

    Stmt* body = new Cond(condition.node(), true_val, false_val);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "B");
  }

  {
    // If both branches are the same then don't do the condition.
    // x ? A[0] = x : A[0] = x => A[0] = x
    VarHandle x("x", kInt);
    BufHandle a("A", {1}, kInt);
    ExprHandle condition(x - x);
    Stmt* true_val = Store::make(a, {0}, x, 1);
    Stmt* false_val = Store::make(a, {0}, x, 1);

    Stmt* body = new Cond(condition.node(), true_val, false_val);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "A");
  }

  {
    // If both branches simplify to the same thing it still works.
    // x ? (x + x) : (2 * x) => x
    VarHandle x("x", kInt);
    BufHandle a("A", {1}, kInt);
    ExprHandle condition(x - x);
    Stmt* true_val = Store::make(a, {0}, ExprHandle(2) * x, 1);
    Stmt* false_val = Store::make(a, {0}, x + x, 1);

    Stmt* body = new Cond(condition.node(), true_val, false_val);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "A");
  }

  {
    // But not if they dont
    // x ? x : (2 * x) => x ? x : (2 * x)
    VarHandle x("x", kInt);
    BufHandle a("A", {1}, kInt);
    ExprHandle condition(x);
    Stmt* true_val = Store::make(a, {0}, x, 1);
    Stmt* false_val = Store::make(a, {0}, ExprHandle(2) * x, 1);

    Stmt* body = new Cond(condition.node(), true_val, false_val);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    ASSERT_EQ(block, nullptr);
  }

  {
    Stmt* cond = new Cond(ExprHandle(false).node(), new Block({}), nullptr);
    Stmt* simplified = IRSimplifier::simplify(cond);
    ASSERT_EQ(simplified, nullptr);
  }

  {
    Stmt* cond = new Cond(ExprHandle(true).node(), nullptr, new Block({}));
    Stmt* simplified = IRSimplifier::simplify(cond);
    ASSERT_EQ(simplified, nullptr);
  }
}

TEST(Simplify, SimplifyEliminateEmptyCond) {
  KernelScope kernel_scope;
  // If the branches are empty in different ways, eliminate.
  {
    VarHandle x("x", kInt);
    ExprHandle condition(x);
    Stmt* true_val = new Block({});

    Stmt* body = new Cond(condition.node(), true_val, nullptr);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    ASSERT_NE(block, nullptr);
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    VarHandle x("x", kInt);
    ExprHandle condition(x);
    Stmt* false_val = new Block({});

    Stmt* body = new Cond(condition.node(), nullptr, false_val);
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    ASSERT_NE(block, nullptr);
    ASSERT_EQ(block->nstmts(), 0);
  }
}

TEST(Simplify, SimplifyConstantComparisons) {
  KernelScope kernel_scope;

  auto ComparisonTest =
      [](ExprHandle a, ExprHandle b, CompareSelectOperation op, int result) {
        ExprHandle body = CompareSelect::make(a, b, op);
        ExprHandle simplified = IRSimplifier::simplify(body);
        IS_IMM_WITH_VAL(Int, simplified.node(), result);
      };

  // Equals.
  ComparisonTest(2, 2, kEQ, 1);
  ComparisonTest(1, 2, kEQ, 0);
  ComparisonTest(2, 1, kEQ, 0);

  // Greater than.
  ComparisonTest(2, 2, kGT, 0);
  ComparisonTest(1, 2, kGT, 0);
  ComparisonTest(2, 1, kGT, 1);

  // Greater or Equal.
  ComparisonTest(2, 2, kGE, 1);
  ComparisonTest(1, 2, kGE, 0);
  ComparisonTest(2, 1, kGE, 1);

  // Less Than.
  ComparisonTest(2, 2, kLT, 0);
  ComparisonTest(1, 2, kLT, 1);
  ComparisonTest(2, 1, kLT, 0);

  // Less or Equal.
  ComparisonTest(2, 2, kLE, 1);
  ComparisonTest(1, 2, kLE, 1);
  ComparisonTest(2, 1, kLE, 0);

  // Not equal.
  ComparisonTest(2, 2, kNE, 0);
  ComparisonTest(1, 2, kNE, 1);
  ComparisonTest(2, 1, kNE, 1);

  // With specified results:
  ExprHandle body = CompareSelect::make(2, 2, 5, 42, kNE);
  ExprHandle simplified = IRSimplifier::simplify(body);
  IS_IMM_WITH_VAL(Int, simplified.node(), 42);
}

TEST(Simplify, SimplifySymbolicComparisons) {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);

  auto TookTrueBranch = [](ExprHandle a) { IS_IMM_WITH_VAL(Int, a.node(), 1); };
  auto TookFalseBranch = [](ExprHandle a) {
    IS_IMM_WITH_VAL(Int, a.node(), 0);
  };

  // EQ

  // x == x => 1
  ExprHandle body = CompareSelect::make(x, x, kEQ);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x == x+1 => 0
  body = CompareSelect::make(x, x + 1, kEQ);
  TookFalseBranch(IRSimplifier::simplify(body));

  // x == x * 2 cannot simplify since we don't know x is nonzero.
  body = CompareSelect::make(x, x * 2, kEQ);
  IS_NODE(CompareSelect, IRSimplifier::simplify(body).node());

  // x == x * 1 => 1
  body = CompareSelect::make(x, x * 1, kEQ);
  TookTrueBranch(IRSimplifier::simplify(body));

  {
    // x == y => x == y
    body = CompareSelect::make(x, y, kEQ);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(CompareSelect, simplified.node(), cmp);
    ASSERT_EQ(cmp->compare_select_op(), kEQ);
    IS_VAR_WITH_NAME(cmp->lhs(), "x");
    IS_VAR_WITH_NAME(cmp->rhs(), "y");
  }

  {
    // x == 5 => x == 5
    body = CompareSelect::make(x, 5, kEQ);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(CompareSelect, simplified.node(), cmp);
    ASSERT_EQ(cmp->compare_select_op(), kEQ);
    IS_VAR_WITH_NAME(cmp->lhs(), "x");
    IS_IMM_WITH_VAL(Int, cmp->rhs(), 5);
  }

  // GT

  // x+1 > x => 1
  body = CompareSelect::make(x + 1, x, kGT);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x > x + 1 => 0
  body = CompareSelect::make(x, x + 1, kGT);
  TookFalseBranch(IRSimplifier::simplify(body));

  // x > x - 1 => 1
  body = CompareSelect::make(x, x - 1, kGT);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x - 1 > x => 0
  body = CompareSelect::make(x - 1, x, kGT);
  TookFalseBranch(IRSimplifier::simplify(body));

  // x > x => 0
  body = CompareSelect::make(x, x, kGT);
  TookFalseBranch(IRSimplifier::simplify(body));

  // x * 2 > x => x * 2 > x
  // since we don't know the sign of x.
  body = CompareSelect::make(x * 2, x, kGT);
  IS_NODE(CompareSelect, IRSimplifier::simplify(body).node());

  // GE

  // x+1 >= x => 1
  body = CompareSelect::make(x + 1, x, kGE);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x >= x + 1 => 0
  body = CompareSelect::make(x, x + 1, kGE);
  TookFalseBranch(IRSimplifier::simplify(body));

  // x >= x => 1
  body = CompareSelect::make(x, x, kGE);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x * 2 >= x => x * 2 >= x
  // since we don't know the sign of x.
  body = CompareSelect::make(x * 2, x, kGE);
  IS_NODE(CompareSelect, IRSimplifier::simplify(body).node());

  // LT

  // x+1 < x => 0
  body = CompareSelect::make(x + 1, x, kLT);
  TookFalseBranch(IRSimplifier::simplify(body));

  // x < x + 1 => 1
  body = CompareSelect::make(x, x + 1, kLT);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x < x => 0
  body = CompareSelect::make(x, x, kLT);
  TookFalseBranch(IRSimplifier::simplify(body));

  // LE

  // x+1 <= x => 0
  body = CompareSelect::make(x + 1, x, kLE);
  TookFalseBranch(IRSimplifier::simplify(body));

  // x <= x + 1 => 1
  body = CompareSelect::make(x, x + 1, kLE);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x <= x => 1
  body = CompareSelect::make(x, x, kLE);
  TookTrueBranch(IRSimplifier::simplify(body));

  // NE

  // x+1 != x => 1
  body = CompareSelect::make(x + 1, x, kNE);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x != x + 1 => 1
  body = CompareSelect::make(x, x + 1, kNE);
  TookTrueBranch(IRSimplifier::simplify(body));

  // x != x => 0
  body = CompareSelect::make(x, x, kNE);
  TookFalseBranch(IRSimplifier::simplify(body));
}

TEST(Simplify, SimplifyEliminateZeroLengthFor) {
  KernelScope kernel_scope;

  {
    // Will eliminate zero loop For.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, 0, 0, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // still works if start is not zero.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, 2, 2, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // works if both terms are variable.
    VarHandle x("x", kInt);
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, x, x, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // works if one term simplifies down.
    VarHandle x("x", kInt);
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, 0, x - x, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    ASSERT_EQ(block->nstmts(), 0);
  }

  {
    // Sanity check does nothing if the condition is not met.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, 0, 3, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE(For, simplified);
  }
}

TEST(Simplify, SimplifyOneLoopFor) {
  KernelScope kernel_scope;

  {
    // Will remove the loop if the body is run once.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, 0, 1, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
  }

  {
    // still works if start is not zero.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, 2, 3, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_IMM_WITH_VAL(Int, store->flat_index(), 2);
  }

  {
    // works if both terms are variable.
    VarHandle x("x", kInt);
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, x, x + 1, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_VAR_WITH_NAME(store->flat_index(), "x");
  }

  {
    // works if one term simplifies down.
    VarHandle x("x", kInt);
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body =
        For::make(i, 0, x - x + 1, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
  }

  {
    // Sanity check does nothing if the condition is not met.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    auto body = For::make(i, 0, 3, Store::make(c, {i}, Load::make(a, {i})));
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE(For, simplified);
  }
}

TEST(Simplify, SimplifyForWontLoseLoopOptions) {
  KernelScope kernel_scope;

  {
    // Sanity check does nothing if the condition is not met.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    LoopOptions options;
    options.set_gpu_block_index(12);
    auto body =
        For::make(i, 0, 1, Store::make(c, {i}, Load::make(a, {i})), options);
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(For, simplified, for_);
    LoopOptions options2 = for_->loop_options();
    ASSERT_EQ(options.gpu_block_index(), options2.gpu_block_index());
  }
}

TEST(Simplify, SimplifyMultilevelFor) {
  KernelScope kernel_scope;

  {
    // Multiple layers of For will be simplified out.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    VarHandle j("j", kInt);
    auto* body = For::make(i, 0, 1, Store::make(c, {i}, Load::make(a, {i})));
    auto outer = For::make(j, 0, 1, body);
    Stmt* simplified = IRSimplifier::simplify(outer);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
  }

  {
    // Will maintain an outer loop if the inner loop is eliminated.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    VarHandle j("j", kInt);
    auto* body = For::make(i, 0, 1, Store::make(c, {i}, Load::make(a, {i})));
    auto outer = For::make(j, 0, 2, body);
    Stmt* simplified = IRSimplifier::simplify(outer);
    For* for__ = static_cast<For*>(simplified);
    IS_NODE_WITH_NAME(For, for__, for_);
    IS_VAR_WITH_NAME(for_->var(), "j");
    IS_IMM_WITH_VAL(Int, for_->start(), 0);
    IS_IMM_WITH_VAL(Int, for_->stop(), 2);
    Block* block = dynamic_cast<Block*>(for_->body());
    ASSERT_NE(block, nullptr);
    IS_NODE_WITH_NAME(Store, block->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_IMM_WITH_VAL(Int, store->flat_index(), 0);
  }

  {
    // Will maintain inner loop if outer loops is eliminated.
    BufHandle a("A", {4}, kInt);
    BufHandle c("C", {4}, kInt);
    VarHandle i("i", kInt);
    VarHandle j("j", kInt);
    auto* body = For::make(i, 0, 2, Store::make(c, {i}, Load::make(a, {i})));
    auto outer = For::make(j, 0, 1, body);
    Stmt* simplified = IRSimplifier::simplify(outer);
    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(For, block->front(), for_);
    IS_VAR_WITH_NAME(for_->var(), "i");
    IS_IMM_WITH_VAL(Int, for_->start(), 0);
    IS_IMM_WITH_VAL(Int, for_->stop(), 2);
    IS_NODE_WITH_NAME(Store, for_->body()->front(), store);
    IS_VAR_WITH_NAME(store->base_handle(), "C");
    IS_VAR_WITH_NAME(store->flat_index(), "i");
  }
}

TEST(Simplify, SimplifyForCleansUp) {
  KernelScope kernel_scope;

  {
    Placeholder a("a", kFloat, {1, 12, 1});
    VarHandle x("x", kInt);
    Tensor* b = Compute(
        "x",
        {{1, "i"}, {12, "m"}, {1, "n"}},
        [](const VarHandle& i, const VarHandle& m, const VarHandle& n) {
          return i + m + n;
        });
    LoopNest l({b});
    l.prepareForCodegen();

    Stmt* body = l.root_stmt();
    Stmt* simplified = IRSimplifier::simplify(body);

    Block* block = dynamic_cast<Block*>(simplified);
    IS_NODE_WITH_NAME(For, block->front(), for_);
    // for is over "m".
    IS_VAR_WITH_NAME(for_->var(), "m");
    // x[m] = m;
    IS_NODE_WITH_NAME(Store, for_->body()->front(), store);
    IS_VAR_WITH_NAME(store->flat_index(), "m");
    IS_VAR_WITH_NAME(store->value(), "m");
  }
}

TEST(Simplify, SimplifyEliminateEmptyFor) {
  KernelScope kernel_scope;

  {
    // Flatten many layers around an empty block to an empty block.
    Stmt* last = new Block({});
    for (int i = 0; i < 11; ++i) {
      VarHandle loopVar("loopVar", kInt);
      last = For::make(loopVar, 0, 10, last);
    }

    Stmt* simplified = IRSimplifier::simplify(last);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 0);
  }
}

TEST(Simplify, SimplifyFlattenBlock) {
  KernelScope kernel_scope;

  {
    // Flatten multiple blocks down to one.
    // { { { stmt1, stmt2 } } } =>  { stmt1, stmt2 }
    BufHandle a("A", {1}, kInt);
    Store* store1 = Store::make(a, {0}, 1, 1);
    Store* store2 = Store::make(a, {0}, 0, 1);

    Block* block1 = new Block({store1, store2});
    Block* block2 = new Block({block1});

    Block* enclosing = new Block({block2});
    Stmt* simplified = IRSimplifier::simplify(enclosing);

    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);

    IS_NODE_WITH_NAME(Store, block->front(), store1_);
    IS_NODE_WITH_NAME(Store, block->back(), store2_);

    ASSERT_EQ(store1->value(), store1_->value());
    ASSERT_EQ(store2->value(), store2_->value());
  }

  {
    // Flatten multiple sub blocks containing statements.
    // { { stmt1 }, { stmt2 } } =>  { stmt1, stmt2 }
    BufHandle a("A", {1}, kInt);
    Store* store1 = Store::make(a, {0}, 1, 1);
    Store* store2 = Store::make(a, {0}, 0, 1);

    Block* block1 = new Block({store1});
    Block* block2 = new Block({store2});

    Block* enclosing = new Block({block1, block2});
    Stmt* simplified = IRSimplifier::simplify(enclosing);

    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);

    IS_NODE_WITH_NAME(Store, block->front(), store1_);
    IS_NODE_WITH_NAME(Store, block->back(), store2_);

    ASSERT_EQ(store1->value(), store1_->value());
    ASSERT_EQ(store2->value(), store2_->value());
  }

  {
    // Flatten sub blocks with different depths.
    // { stmt1 , { { stmt2 } } } =>  { stmt1, stmt2 }
    BufHandle a("A", {1}, kInt);
    Store* store1 = Store::make(a, {0}, 1, 1);
    Store* store2 = Store::make(a, {0}, 0, 1);

    Block* block1 = new Block({store2});
    Block* block2 = new Block({block1});

    Block* enclosing = new Block({store1, block2});
    Stmt* simplified = IRSimplifier::simplify(enclosing);

    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);

    IS_NODE_WITH_NAME(Store, block->front(), store1_);
    IS_NODE_WITH_NAME(Store, block->back(), store2_);

    ASSERT_EQ(store1->value(), store1_->value());
    ASSERT_EQ(store2->value(), store2_->value());
  }

  {
    // Flatten many layers around an empty block to an empty block.
    Stmt* last = new Block({});
    for (int i = 0; i < 11; ++i) {
      last = new Block({last});
    }

    Stmt* simplified = IRSimplifier::simplify(last);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 0);
  }
}

TEST(Simplify, SimplifyEliminateZeroLengthAlloc) {
  KernelScope kernel_scope;

  {
    // Simple positive case.
    BufHandle b("x", {0}, kInt);

    Allocate* alloc = Allocate::make(b);
    Free* free_ = Free::make(b);

    Block* block1 = new Block({alloc, free_});
    ASSERT_EQ(block1->nstmts(), 2);

    Stmt* simplified = IRSimplifier::simplify(block1);
    IS_NODE_WITH_NAME(Block, simplified, block2);
    ASSERT_EQ(block2->nstmts(), 0);
  }

  {
    // Simple negative case.
    BufHandle b("x", {2}, kInt);

    Allocate* alloc = Allocate::make(b);
    Free* free_ = Free::make(b);

    Block* block1 = new Block({alloc, free_});
    ASSERT_EQ(block1->nstmts(), 2);

    Stmt* simplified = IRSimplifier::simplify(block1);
    IS_NODE_WITH_NAME(Block, simplified, block2);
    ASSERT_EQ(block2->nstmts(), 2);
  }

  {
    // Finds right Alloc/Free.
    BufHandle b1("x", {0}, kInt);
    BufHandle b2("y", {2}, kInt);

    Allocate* alloc1 = Allocate::make(b1);
    Allocate* alloc2 = Allocate::make(b2);
    Free* free2_ = Free::make(b2);
    Free* free1_ = Free::make(b1);

    Block* block1 = new Block({alloc1, alloc2, free2_, free1_});
    ASSERT_EQ(block1->nstmts(), 4);

    Stmt* simplified = IRSimplifier::simplify(block1);
    IS_NODE_WITH_NAME(Block, simplified, block2);
    ASSERT_EQ(block2->nstmts(), 2);
    IS_NODE_WITH_NAME(Allocate, block2->stmts().front(), simplified_alloc);
    IS_VAR_WITH_NAME(simplified_alloc->buffer_var(), "y");
    IS_NODE_WITH_NAME(Free, block2->stmts().back(), simplified_free);
    ASSERT_EQ(simplified_alloc->buffer_var(), simplified_free->buffer_var());
  }

  {
    // Dynamic shape.
    VarHandle z("z", kInt);
    BufHandle b1("x", {0}, kInt);
    BufHandle b2("y", {z}, kInt);

    Allocate* alloc1 = Allocate::make(b1);
    Allocate* alloc2 = Allocate::make(b2);
    Free* free2_ = Free::make(b2);
    Free* free1_ = Free::make(b1);

    Block* block1 = new Block({alloc1, alloc2, free2_, free1_});
    ASSERT_EQ(block1->nstmts(), 4);
    Stmt* simplified = IRSimplifier::simplify(block1);
    IS_NODE_WITH_NAME(Block, simplified, block2);
    ASSERT_EQ(block2->nstmts(), 2);
  }
}

TEST(Simplify, DontSimplifyRand) {
  KernelScope kernel_scope;

  {
    // rand() + rand() = rand() + rand() NOT 2 * rand().
    ExprHandle body =
        Intrinsics::make(kRand, kInt) + Intrinsics::make(kRand, kInt);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Add, simplified.node(), add);
    IS_RAND(add->lhs());
    IS_RAND(add->rhs());
  }

  {
    // rand() - rand() = rand() - rand() NOT 0.
    ExprHandle body =
        Intrinsics::make(kRand, kFloat) - Intrinsics::make(kRand, kFloat);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Sub, simplified.node(), sub);
    IS_RAND(sub->lhs());
    IS_RAND(sub->rhs());
  }

  {
    // rand() * rand() = rand() * rand().
    ExprHandle body =
        Intrinsics::make(kRand, kInt) * Intrinsics::make(kRand, kInt);
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_RAND(mul->lhs());
    IS_RAND(mul->rhs());
  }
}

TEST(Simplify, SimplifyReorderForCond) {
  KernelScope kernel_scope;
  BufHandle a("A", {4}, kInt);
  BufHandle b("B", {1}, kInt);
  BufHandle c("C", {4}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);

  {
    // for ( if ( ... ) ) => if ( for ( ... ) ).
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(j, 10, CompareSelectOperation::kLT),
            Store::make(c, {i}, Load::make(a, {i})),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
  }

  {
    // Can't reorder if condition is dependent on the loop var.
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(c, {i}, Load::make(a, {i})),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(For, simplified, loop);
    IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
  }

  {
    // Can't reorder if condition is dependent on a var that is modified inside
    // the loop.
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(c, {0}), 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(For, simplified, loop);
    IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
  }

  {
    // Condition based on buffer not referenced in body. Can reorder here.
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(b, {0}), 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
  }

  {
    // Condition based on buffer read only in body. Can reorder here.
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(a, {0}), 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
  }

  {
    // Condition depends on Let in the loop. Cannot reorder.
    auto body = For::make(
        i,
        0,
        4,
        Block::make(
            {Let::make(j, 3),
             Cond::make(
                 CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                 Store::make(c, {0}, Load::make(a, {i})),
                 nullptr)}));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(For, simplified, loop);
    IS_NODE_WITH_NAME(Let, loop->body()->front(), let);
    IS_NODE_WITH_NAME(Cond, loop->body()->back(), cond);
  }

  {
    // Multi level Ifs where all conditions are distinct. Move BOTH Cond
    // statements outside the loop.
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(a, {0}), 10, CompareSelectOperation::kLT),
            Cond::make(
                CompareSelect::make(j, 10, CompareSelectOperation::kEQ),
                Store::make(c, {0}, Load::make(a, {i})),
                nullptr),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    IS_NODE_WITH_NAME(Cond, true_block->front(), cond2);
    IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_block2);
    IS_NODE_WITH_NAME(For, true_block2->front(), loop);
  }

  {
    // Multi level Ifs where the inner condition does depend on a loop var,
    // reorder only the first Cond.
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(a, {0}), 10, CompareSelectOperation::kLT),
            Cond::make(
                CompareSelect::make(i, 10, CompareSelectOperation::kEQ),
                Store::make(c, {0}, Load::make(a, {i})),
                nullptr),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
    IS_NODE_WITH_NAME(Block, loop->body(), loop_body);
    IS_NODE_WITH_NAME(Cond, loop_body->front(), cond2);
  }

  {
    // Don't reorder if there's an else block of the Cond.
    // We could, but is it much better?
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(j, 10, CompareSelectOperation::kLT),
            Store::make(c, {0}, Load::make(a, {i})),
            Store::make(c, {0}, 0)));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(For, simplified, loop);
    IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
  }

  {
    // Condition uses distinct region of Tensor.
    // We could reorder here wih better analysis, but we don't. Included for
    // completeness.
    auto body = For::make(
        i,
        0,
        4,
        Cond::make(
            CompareSelect::make(
                Load::make(c, {0}), 10, CompareSelectOperation::kLT),
            Store::make(c, {1}, Load::make(a, {i})),
            nullptr));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(For, simplified, loop);
    IS_NODE_WITH_NAME(Cond, loop->body()->front(), cond);
  }
}

TEST(Simplify, SimplifyFuseConditions) {
  KernelScope kernel_scope;
  BufHandle a("A", {2}, kInt);
  BufHandle b("B", {2}, kInt);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);

  {
    // Can fuse since the conditions are identical.
    // if (A) { X }; if (A) { Y }; => if (A) { X; Y }
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             nullptr),
         Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {1}, i),
             nullptr)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 2);
    ASSERT_EQ(cond->false_stmt(), nullptr);
  }

  {
    // Can't fuse, conditions are not identical in lhs (i != j).
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             nullptr),
         Cond::make(
             CompareSelect::make(j, 10, CompareSelectOperation::kLT),
             Store::make(a, {1}, i),
             nullptr)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);
    IS_NODE_WITH_NAME(Cond, block->front(), cond1);
    IS_NODE_WITH_NAME(Cond, block->back(), cond2);

    IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);
    IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);
    ASSERT_EQ(true_stmt1->nstmts(), 1);
    ASSERT_EQ(true_stmt2->nstmts(), 1);

    ASSERT_EQ(cond1->false_stmt(), nullptr);
    ASSERT_EQ(cond2->false_stmt(), nullptr);
  }
  {
    // Can't fuse, conditions are not identical in rhs (10 != 11).
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             nullptr),
         Cond::make(
             CompareSelect::make(i, 11, CompareSelectOperation::kLT),
             Store::make(a, {1}, i),
             nullptr)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);
    IS_NODE_WITH_NAME(Cond, block->front(), cond1);
    IS_NODE_WITH_NAME(Cond, block->back(), cond2);

    IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);
    IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);
    ASSERT_EQ(true_stmt1->nstmts(), 1);
    ASSERT_EQ(true_stmt2->nstmts(), 1);

    ASSERT_EQ(cond1->false_stmt(), nullptr);
    ASSERT_EQ(cond2->false_stmt(), nullptr);
  }

  {
    // Can't fuse, conditions are not identical in operation (LT vs GT).
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             nullptr),
         Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kGT),
             Store::make(a, {1}, i),
             nullptr)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);
    IS_NODE_WITH_NAME(Cond, block->front(), cond1);
    IS_NODE_WITH_NAME(Cond, block->back(), cond2);

    IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);
    IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);
    ASSERT_EQ(true_stmt1->nstmts(), 1);
    ASSERT_EQ(true_stmt2->nstmts(), 1);

    ASSERT_EQ(cond1->false_stmt(), nullptr);
    ASSERT_EQ(cond2->false_stmt(), nullptr);
  }

  {
    // Can't fuse, CompareSelect results are different.
    // Actually we totally could if we normalized CompareSelect results, but
    // TODO for later.
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(
                 i,
                 10,
                 new IntImm(1),
                 new IntImm(0),
                 CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             nullptr),
         Cond::make(
             CompareSelect::make(
                 j,
                 10,
                 new IntImm(2),
                 new IntImm(0),
                 CompareSelectOperation::kLT),
             Store::make(a, {1}, i),
             nullptr)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);
    IS_NODE_WITH_NAME(Cond, block->front(), cond1);
    IS_NODE_WITH_NAME(Cond, block->back(), cond2);

    IS_NODE_WITH_NAME(Block, cond1->true_stmt(), true_stmt1);
    IS_NODE_WITH_NAME(Block, cond2->true_stmt(), true_stmt2);
    ASSERT_EQ(true_stmt1->nstmts(), 1);
    ASSERT_EQ(true_stmt2->nstmts(), 1);

    ASSERT_EQ(cond1->false_stmt(), nullptr);
    ASSERT_EQ(cond2->false_stmt(), nullptr);
  }

  {
    // Can fuse with false stmt only.
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             nullptr,
             Store::make(a, {0}, i)),
         Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             nullptr,
             Store::make(a, {1}, i))});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->false_stmt(), false_stmt);
    ASSERT_EQ(false_stmt->nstmts(), 2);
    ASSERT_EQ(cond->true_stmt(), nullptr);
  }

  {
    // Can fuse with both true and false stmt.
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             Store::make(b, {0}, i)),
         Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {1}, i),
             Store::make(b, {1}, i))});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 2);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), false_stmt);
    ASSERT_EQ(false_stmt->nstmts(), 2);
  }

  {
    // Can fuse with mismatched true / false stmt existing
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             nullptr),
         Cond::make(
             CompareSelect::make(i, 10, CompareSelectOperation::kLT),
             nullptr,
             Store::make(b, {1}, i))});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 1);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), false_stmt);
    ASSERT_EQ(false_stmt->nstmts(), 1);
  }

  {
    // Can fuse partial block contents, ie when there are non fused stmts before
    // and after.
    // before:
    // if (j < 10) { A[0] = j; }
    // if (i < 10) { A[0] = i; }
    // if (i < 10) { A[1] = i; }
    // if (i < 11) { A[1] = j; }
    //
    // after:
    //
    // if (j < 10) { A[0] = j; }
    // if (i < 10) {
    //   A[0] = i;
    //   A[1] = i;
    // }
    // if (i < 11) { A[1] = j; }

    auto body = Block::make({
        Cond::make(
            CompareSelect::make(j, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, j),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, i),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {1}, i),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 11, CompareSelectOperation::kLT),
            Store::make(a, {1}, j),
            nullptr),
    });
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 3);
    auto it = block->begin();
    it++;
    IS_NODE_WITH_NAME(Cond, *it, cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 2);
    ASSERT_EQ(cond->false_stmt(), nullptr);
  }

  {
    // Can fuse longer sequences of identical conditions.
    auto body = Block::make({
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, j),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, i),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {1}, i),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {1}, j),
            nullptr),
    });
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 4);
    ASSERT_EQ(cond->false_stmt(), nullptr);
  }

  {
    // Can't fuse through a non condition.
    auto body = Block::make({
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, j),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {0}, i),
            nullptr),
        Store::make(b, {1}, i + j),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {1}, i),
            nullptr),
        Cond::make(
            CompareSelect::make(i, 10, CompareSelectOperation::kLT),
            Store::make(a, {1}, j),
            nullptr),
    });
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 3);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 2);
    ASSERT_EQ(cond->false_stmt(), nullptr);

    IS_NODE_WITH_NAME(Cond, block->back(), cond2);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt2);
    ASSERT_EQ(true_stmt2->nstmts(), 2);
    ASSERT_EQ(cond2->false_stmt(), nullptr);

    auto it = block->begin();
    it++;
    IS_NODE_WITH_NAME(Store, *it, middle);
  }

  {
    // Can fuse if the conditions simplify to the same thing.
    auto body = Block::make(
        {Cond::make(
             CompareSelect::make(
                 i * 2,
                 ExprHandle(87) % ExprHandle(11),
                 CompareSelectOperation::kLT),
             Store::make(a, {0}, i),
             nullptr),
         Cond::make(
             CompareSelect::make(
                 i * 2,
                 ExprHandle(300) / ExprHandle(30),
                 CompareSelectOperation::kLT),
             Store::make(a, {1}, i),
             nullptr)});
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 2);
    ASSERT_EQ(cond->false_stmt(), nullptr);
  }

  {
    // Can fuse non-CompareSelects.
    // if (i) { X } if (i) { Y } => if (i) { X; Y }
    auto body = Block::make(
        {Cond::make(i, Store::make(a, {0}, i), nullptr),
         Cond::make(i, Store::make(a, {1}, i), nullptr)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    IS_NODE_WITH_NAME(Cond, block->front(), cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_stmt);
    ASSERT_EQ(true_stmt->nstmts(), 2);
    ASSERT_EQ(cond->false_stmt(), nullptr);
  }

  {
    // Sanity check wont fuse different non-CompareSelects.
    auto body = Block::make(
        {Cond::make(i, Store::make(a, {0}, i), nullptr),
         Cond::make(j, Store::make(a, {1}, i), nullptr)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);
    IS_NODE_WITH_NAME(Cond, block->front(), cond1);
    IS_NODE_WITH_NAME(Cond, block->back(), cond2);
  }

  {
    // Sanity check constant condition elimination still occurs when merging is
    // possible.
    auto body = Block::make(
        {Cond::make(1, Store::make(a, {0}, i), nullptr),
         Cond::make(1, Store::make(a, {1}, i), nullptr)});
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 2);
    IS_NODE_WITH_NAME(Store, block->front(), store1);
    IS_NODE_WITH_NAME(Store, block->back(), store2);
  }

  {
    // Sanity check for-cond reordering occurs after fusing.
    auto body = For::make(
        i,
        0,
        4,
        Block::make(
            {Cond::make(
                 CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                 Store::make(a, {1}, Load::make(b, {0})),
                 nullptr),
             Cond::make(
                 CompareSelect::make(j, 10, CompareSelectOperation::kLT),
                 Store::make(a, {2}, Load::make(b, {0})),
                 nullptr)}));

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Cond, simplified, cond);
    IS_NODE_WITH_NAME(Block, cond->true_stmt(), true_block);
    IS_NODE_WITH_NAME(For, true_block->front(), loop);
  }
}

TEST(Simplify, SimplifySyncThreads) {
  KernelScope kernel_scope;
  BufHandle a("A", {4}, kInt);
  VarHandle i("i", kInt);

  {
    // Merge two inner SyncThreads.
    auto body = Block::make(
        {Store::make(a, {0}, 1, 1),
         new SyncThreads(),
         new SyncThreads(),
         Store::make(a, {1}, 0, 1)});
    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 3);
    auto it = block->begin();
    IS_NODE(Store, *it++);
    IS_NODE(SyncThreads, *it++);
    IS_NODE(Store, *it++);
  }

  {
    // Eliminate outer SyncThreads.
    auto body = Block::make(
        {new SyncThreads(), Store::make(a, {1}, 0, 1), new SyncThreads()});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    auto it = block->begin();
    IS_NODE(Store, *it);
  }

  {
    // Merge many inner SyncThreads.
    auto body = Block::make(
        {Store::make(a, {0}, 1, 1),
         new SyncThreads(),
         new SyncThreads(),
         new SyncThreads(),
         new SyncThreads(),
         new SyncThreads(),
         Store::make(a, {1}, 0, 1)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 3);
    auto it = block->begin();
    IS_NODE(Store, *it++);
    IS_NODE(SyncThreads, *it++);
    IS_NODE(Store, *it++);
  }

  {
    // Merge multiple outer SyncThreads.
    auto body = Block::make(
        {new SyncThreads(),
         new SyncThreads(),
         Store::make(a, {1}, 0, 1),
         new SyncThreads(),
         new SyncThreads(),
         new SyncThreads(),
         new SyncThreads()});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 1);
    auto it = block->begin();
    IS_NODE(Store, *it);
  }

  {
    // Merge multiple sections;
    auto body = Block::make(
        {Store::make(a, {0}, 1, 1),
         new SyncThreads(),
         new SyncThreads(),
         Store::make(a, {1}, 0, 1),
         Store::make(a, {2}, 0, 1),
         new SyncThreads(),
         new SyncThreads(),
         new SyncThreads(),
         Store::make(a, {3}, 0, 1)});

    Stmt* simplified = IRSimplifier::simplify(body);
    IS_NODE_WITH_NAME(Block, simplified, block);
    ASSERT_EQ(block->nstmts(), 6);
    auto it = block->begin();
    IS_NODE(Store, *it++);
    IS_NODE(SyncThreads, *it++);
    IS_NODE(Store, *it++);
    IS_NODE(Store, *it++);
    IS_NODE(SyncThreads, *it++);
    IS_NODE(Store, *it++);
  }
}

TEST(Simplify, SimplifyRampSubBroadcast) {
  KernelScope kernel_scope;
  int num_lanes = 4;
  ExprHandle ramp = Ramp::make(ExprHandle(0), ExprHandle(6), num_lanes);
  ExprHandle broadcast = Broadcast::make(ExprHandle(-5), num_lanes);
  ExprHandle simplified = IRSimplifier::simplify(ramp - broadcast);
  Ramp* newRamp = simplified.AsNode<Ramp>();
  IS_NODE_WITH_NAME(IntImm, newRamp->base(), base);
  ASSERT_EQ(base->value(), 5);
  IS_NODE_WITH_NAME(IntImm, newRamp->stride(), stride);
  ASSERT_EQ(stride->value(), 6);
  ASSERT_EQ(newRamp->lanes(), num_lanes);
}

TEST(Simplify, SimplifyBroadcastTermExpander) {
  KernelScope kernel_scope;
  int num_lanes = 8;
  ExprHandle bc0 = Broadcast::make(ExprHandle(0), num_lanes);
  ExprHandle bc1 = Broadcast::make(ExprHandle(1), num_lanes);
  ExprHandle bc2 = Broadcast::make(ExprHandle(2), num_lanes);
  // NB: We need a term in the middle which isn't simplified to trigger the
  // relevant path in TermExpander::mutate. The two bc1 terms are brought
  // together and simplified to 2 * bc1, which then needs to make 2 multi-lane.
  ExprHandle simplified = IRSimplifier::simplify(bc1 + (bc0 / bc2) + bc1);
  BufHandle buf("buf", {num_lanes}, kInt);
  // The result isn't fully simplified currently and thus would be brittle to
  // match. Observe its value instead.
  auto store = Store::make(
      buf,
      {Ramp::make(0, 1, num_lanes)},
      simplified,
      Broadcast::make(ExprHandle(1), num_lanes));
  SimpleIREvaluator eval(store, {buf});
  std::vector<int> output(num_lanes);
  eval(output);
  for (int i = 0; i < num_lanes; ++i) {
    ASSERT_EQ(output[i], 2);
  }
}

TEST(Simplify, DISABLED_CompareSelectCondAlwaysInLoopBounds) {
  // Before:
  //   for (int n = 1; n < N; n++) {
  //     b[n] = n < 1 ? 0.f : 1.f;
  //   }
  // After:
  //   for (int n = 1; n < N; n++) {
  //     b[n] = 1.f;
  //   }
  KernelScope kernel_scope;
  constexpr int N = 8;
  Placeholder b("b", kFloat, {N});
  VarHandle n("n", kInt);
  Stmt* s = For::make(
      n, 1, N, b.store({n}, CompareSelect::make(n, 1, 0.f, 1.0f, kLT)));
  s = IRSimplifier::simplify(s);
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: b[n] = 1.f;
)IR",
      oss.str());
}

TEST(Simplify, DISABLED_IfThenCondAlwaysInLoopBounds) {
  // Before:
  //   for (int n = 1; n < N; n++) {
  //     b[n] = IfThenElse(n < 1 ? 1 : 0, 0.f, 1.f);
  //   }
  // After:
  //   for (int n = 1; n < N; n++) {
  //     b[n] = 1.f;
  //   }
  KernelScope kernel_scope;
  constexpr int N = 8;
  Placeholder b("b", kFloat, {N});
  VarHandle n("n", kInt);
  Stmt* s =
      For::make(n, 1, N, b.store({n}, IfThenElse::make(n < 1, 0.f, 1.0f)));
  s = IRSimplifier::simplify(s);
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: b[n] = 1.f;
)IR",
      oss.str());
}

TEST(Simplify, DISABLED_MultiClauseCondAlwaysInLoopBounds) {
  // This test mimics the unpadded region of a conv2d.  We want to remove any
  // conditional that is provably satisfied (or unsatisfied) by the entire loop
  // range.
  // Before:
  //   for (int i = 1; i < 7; i++) {
  //     for (int j = 1; j < 7; j++) {
  //       b[i, j] = IfThenElse(
  //         j>=7 ? 1 : (i>=7 ? 1 : (j<1 ? 1 : (i<1 ? 1 : 0))), 0.f, 1.f);
  // After:
  //   for (int i = 1; i < 7; i++) {
  //     for (int j = 1; j < 7; j++) {
  //       b[i, j] = 1.f;
  KernelScope kernel_scope;
  constexpr int N = 8;
  Placeholder b("b", kFloat, {N, N});
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto csel = CompareSelect::make(i, 1, kLT);
  csel = CompareSelect::make(j, 1, 1, csel, kLT);
  csel = CompareSelect::make(i, N - 1, 1, csel, kGE);
  csel = CompareSelect::make(j, N - 1, 1, csel, kGE);
  Stmt* s = b.store({i, j}, IfThenElse::make(csel, 0.f, 1.0f));
  s = For::make(j, 1, N - 1, s);
  s = For::make(i, 1, N - 1, s);
  s = IRSimplifier::simplify(s);
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: b[n] = 1.f;
)IR",
      oss.str());
}

TEST(Simplify, DISABLED_SimplifyLoopBounds) {
  // This test mimics the padded region of a conv2d.  We want to adjust the
  // loop bounds such that the condition will be always met.  Note that this
  // could be solved by peeling, and applying the range-based conditional
  // simplification in the previous tests.
  // Before:
  //   for (int i = 0; i < 3; i++) {
  //     for (int j = 0; j < 3; j++) {
  //       b[i, j] = (b[i, j]) + (IfThenElse(
  //         j>=7 ? 1 : (i>=7 ? 1 : (j<1 ? 1 : (i<1 ? 1 : 0))), 0.f, a[i, j]));
  // After:
  //   for (int i = 1; i < 3; i++) {
  //     for (int j = 1; j < 3; j++) {
  //       b[i, j] = (b[i, j]) + 1.f;
  KernelScope kernel_scope;
  constexpr int N = 8;
  constexpr int K = 3;
  Placeholder a("a", kFloat, {N, N});
  Placeholder b("b", kFloat, {N, N});
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);
  auto csel = CompareSelect::make(i, 1, kLT);
  csel = CompareSelect::make(j, 1, 1, csel, kLT);
  csel = CompareSelect::make(i, N - 1, 1, csel, kGE);
  csel = CompareSelect::make(j, N - 1, 1, csel, kGE);
  Stmt* s = b.store(
      {i, j}, b.load({i, j}) + IfThenElse::make(csel, 0.f, a.load({i, j})));
  s = For::make(j, 0, K, s);
  s = For::make(i, 0, K, s);
  s = IRSimplifier::simplify(s);
  std::ostringstream oss;
  oss << *s;
  torch::jit::testing::FileCheck().run(
      R"IR(
# CHECK: for (int i = 1; i < 3; i++) {
# CHECK: for (int j = 1; j < 3; j++) {
# CHECK-NOT: IfThenElse
)IR",
      oss.str());
}

} // namespace jit
} // namespace torch

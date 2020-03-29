#include "test/cpp/tensorexpr/test_base.h"

#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/hash_provider.h"
#include "torch/csrc/jit/tensorexpr/ir_simplifier.h"
#include "torch/csrc/jit/tensorexpr/llvm_codegen.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"

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

void testConstantFoldSimple() {
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

void testConstantFoldTwoLayer() {
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

void testConstantFoldShifts() {
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

void testConstantFoldBitwise() {
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
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);

  SimpleIRExprEval eval(newF);
  SimpleIRExprEval ref(fn);

  ASSERT_EQ(eval.value<float>(), ref.value<float>());
}

void testConstantFoldMinMax() {
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
  ASSERT_NE(newF.AsNode<FloatImm>(), nullptr);
  ASSERT_EQ(newF.AsNode<FloatImm>()->value(), 1);

  SimpleIRExprEval eval(newF);
  SimpleIRExprEval ref(fn);

  ASSERT_EQ(eval.value<float>(), ref.value<float>());
}

void testConstantFoldWithVar() {
  KernelScope kernel_scope;
  {
    VarHandle x("x", kInt);
    ExprHandle body = x * (ExprHandle(2) + ExprHandle(4));

    ExprHandle newF = IRSimplifier::simplify(body);
    const Mul* root = newF.AsNode<Mul>();
    ASSERT_NE(root, nullptr);
    ASSERT_NE(dynamic_cast<const IntImm*>(root->lhs()), nullptr);

    ExprHandle result = Let::make(x, ExprHandle(3), newF);
    SimpleIRExprEval eval(result);
    ASSERT_EQ(eval.value<int>(), 3 * (2 + 4));
  }

  {
    VarHandle x("x", kFloat);
    ExprHandle body = x * (ExprHandle(2.f) + ExprHandle(4.f));

    ExprHandle newF = IRSimplifier::simplify(body);
    const Mul* root = newF.AsNode<Mul>();
    ASSERT_NE(root, nullptr);
    ASSERT_NE(dynamic_cast<const FloatImm*>(root->rhs()), nullptr);

    ExprHandle result = Let::make(x, ExprHandle(3.f), newF);
    SimpleIRExprEval eval(result);
    ASSERT_EQ(eval.value<float>(), 3 * (2 + 4));
  }
}

void testUnFoldableExpr() {
  KernelScope kernel_scope;
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle body = (ExprHandle(3) * x) + (ExprHandle(5) * y);

  ExprHandle newF = IRSimplifier::simplify(body);
  const Add* root = newF.AsNode<Add>();
  ASSERT_NE(root, nullptr);
  ASSERT_EQ(dynamic_cast<const FloatImm*>(root->lhs()), nullptr);
  ASSERT_EQ(dynamic_cast<const FloatImm*>(root->rhs()), nullptr);

  ExprHandle result = Let::make(x, ExprHandle(3.f), newF);
  result = Let::make(y, ExprHandle(2.f), result);
  SimpleIRExprEval eval(result);
  ASSERT_EQ(eval.value<float>(), 9 + 10);
}

// bool operator==(
//     const torch::jit::tensorexpr::SimplifierHashType& a,
//     const torch::jit::tensorexpr::SimplifierHashType& b) {
//   return a._h == b._h;
// }

// bool operator==(
//     const torch::jit::tensorexpr::SimplifierHashType& a,
//     const size_t s) {
//   return a._h == s;
// }

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

  ASSERT_NE(hash_x, (size_t)0);
  ASSERT_NE(hash_a, (size_t)0);
  ASSERT_NE(hash_f, (size_t)0);
  ASSERT_NE(hash_x, hash_a);
  ASSERT_NE(hash_x, hash_f);
  ASSERT_NE(hash_a, hash_f);
}

void testHashEquivalence() {
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

void testHashEquivalenceAfterFolding() {
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
  ASSERT_TRUE(hasher.cachedHash(memcpy_stmt));
  auto hash_t = hasher.hash(memcpy_stmt);
  ASSERT_TRUE(hasher.cachedHash(store_ramp_stmt));
  auto hash_f = hasher.hash(store_ramp_stmt);

  // Root not equal to either branch, and branches not equal.
  ASSERT_NE(hash_r, hash_t);
  ASSERT_NE(hash_r, hash_f);
  ASSERT_NE(hash_t, hash_f);
}

/// (2 + x) + 4 => x + 6
void testSimplifyAdd() {
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
void testSimplifySub() {
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

/// 2 * (1 - x) - 4 => -2 * (x + 3)
void testSimplifyMultiLayer() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body = ExprHandle(2) * ((ExprHandle(1) - x) - ExprHandle(4));
  ExprHandle simplified = IRSimplifier::simplify(body);
  IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
  IS_IMM_WITH_VAL(Int, mul->lhs(), -2);
  IS_NODE_WITH_NAME(Add, mul->rhs(), add);
  IS_VAR_WITH_NAME(add->lhs(), "x");
  IS_IMM_WITH_VAL(Int, add->rhs(), 3);
}

/// 2 * (3 * x) - (x * 4) => 2 * x
void testSimplifyMultiTerm() {
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
void testSimplifyCasts() {
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
void testSimplifyEliminatesNoOps() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  ExprHandle body = (x + ExprHandle(0)) * 1;

  ExprHandle simplified = IRSimplifier::simplify(body);
  const Var* root = simplified.AsNode<Var>();
  ASSERT_NE(root, nullptr);
  ASSERT_EQ(root->name_hint(), "x");
}

/// Cannot simplify this.
void testSimplifyMultiVar() {
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
void testSimplifyReorderings() {
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
void testSimplifyEliminatesVar() {
  KernelScope kernel_scope;
  VarHandle x("x", kInt);
  VarHandle y("y", kInt);
  ExprHandle body = y + x * ExprHandle(0);

  ExprHandle simplified = IRSimplifier::simplify(body);
  IS_VAR_WITH_NAME(simplified.node(), "y");
}

void testSimplifyAdds() {
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
    // (x - y) + (x - y) => -2 * (y - x)
    ExprHandle body = (x - y) + (x - y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), -2);

    IS_NODE_WITH_NAME(Sub, mul->rhs(), rhs);
    IS_VAR_WITH_NAME(rhs->lhs(), "y");
    IS_VAR_WITH_NAME(rhs->rhs(), "x");
  }
}

void testSimplifyMuls() {
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
void testSimplifySubs() {
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
    // (x + y) - 2 * (x + y) => -1 * (x + y)
    ExprHandle body = (x + y) - ExprHandle(2) * (x + y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), -1);
    IS_NODE_WITH_NAME(Add, mul->rhs(), add);
    IS_VAR_WITH_NAME(add->lhs(), "y");
    IS_VAR_WITH_NAME(add->rhs(), "x");
  }

  {
    // (x + y) - y => x
    ExprHandle body = (x + y) - y;
    ExprHandle simplified = IRSimplifier::simplify(body);
    IS_VAR_WITH_NAME(simplified.node(), "x");
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
}

// Test that mixing ops together simplifies as expected.
void testSimplifyMultiOp() {
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
}

// Test that chaining many ops together works as expected.
void testSimplifyManyOps() {
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

void testSimplifyFactorization() {
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
    // (-2 * x) + (4 * y) => -2 * (x - 2 * y)
    ExprHandle body = (ExprHandle(-2) * x + ExprHandle(4) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), -2);

    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    IS_VAR_WITH_NAME(sub->lhs(), "x");
    IS_NODE_WITH_NAME(Mul, sub->rhs(), mul2);
    IS_IMM_WITH_VAL(Int, mul2->lhs(), 2);
    IS_VAR_WITH_NAME(mul2->rhs(), "y");
  }
}

// (4 * x + y + z * 2) + (4 * x + y + z * 4) => 2 * (y + 3 * z + 4 * x)
void testSimplifyFactorizeUneven() {
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
void testSimplifyDeeperTerms() {
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
void testSimplifyDeeperDifference() {
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
void testSimplifyFoldComplexDifference() {
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

void testSimplifyIfComponents() {
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

void testSimplifyOpaqueTerms() {
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

void testSimplifyWontReorderFloat() {
  KernelScope kernel_scope;

  {
    // 3 * (3 * x) - 3 * (3 * y) => -9 * (y - x)
    // This is an expression we can simplify.
    VarHandle x("x", kInt);
    VarHandle y("y", kInt);

    ExprHandle body = ExprHandle(3) * (ExprHandle(3) * x) -
        ExprHandle(3) * (ExprHandle(3) * y);
    ExprHandle simplified = IRSimplifier::simplify(body);

    IS_NODE_WITH_NAME(Mul, simplified.node(), mul);
    IS_IMM_WITH_VAL(Int, mul->lhs(), -9);
    IS_NODE_WITH_NAME(Sub, mul->rhs(), sub);
    IS_VAR_WITH_NAME(sub->lhs(), "y");
    IS_VAR_WITH_NAME(sub->rhs(), "x");
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

} // namespace jit
} // namespace torch

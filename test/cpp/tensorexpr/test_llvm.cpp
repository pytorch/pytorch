#ifdef TORCH_ENABLE_LLVM
#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <test/cpp/tensorexpr/padded_buffer.h>
#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <cmath>
#include <numeric>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

using LLVMExprEval = ExprEval<LLVMCodeGen>;

// Typed tests, can't use gtest params here due to the way we instantiate tests.
#define TEST_LLVM_SCALAR_TYPES(_) \
  _(uint8_t, Byte, 24)            \
  _(int8_t, Char, -20)            \
  _(int16_t, Short, 3332)         \
  _(int, Int, 123456)             \
  _(int64_t, Long, 2631563121321) \
  _(float, Float, 0.122)          \
  _(double, Double, 0.21312)      \
  _(at::Half, Half, 0.128f)

#define IMM_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##ImmTest) {                      \
    KernelScope kernel_scope;                      \
    auto a = Name##Imm::make(Val);                 \
    LLVMExprEval cg(a);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val, 0.1);     \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val);            \
    }                                              \
  }
TEST_LLVM_SCALAR_TYPES(IMM_TEST)
#undef IMM_TEST

#define ADD_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##AddTest) {                      \
    KernelScope kernel_scope;                      \
    auto a = Name##Imm::make(Val);                 \
    auto b = Name##Imm::make(Val * 2);             \
    auto c = Add::make(a, b);                      \
    LLVMExprEval cg(c);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val * 3, 0.1); \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val * 3);        \
    }                                              \
  }
TEST_LLVM_SCALAR_TYPES(ADD_TEST)
#undef ADD_TEST

#define SUB_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##SubTest) {                      \
    KernelScope kernel_scope;                      \
    auto a = Name##Imm::make(Val * 2);             \
    auto b = Name##Imm::make(Val);                 \
    auto c = Sub::make(a, b);                      \
    LLVMExprEval cg(c);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val, 0.1);     \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val);            \
    }                                              \
  }
TEST_LLVM_SCALAR_TYPES(SUB_TEST)
#undef SUB_TEST

#define MUL_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##MulTest) {                      \
    KernelScope kernel_scope;                      \
    auto a = Name##Imm::make(Val);                 \
    auto b = Name##Imm::make((Type)4);             \
    auto c = Mul::make(a, b);                      \
    LLVMExprEval cg(c);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), Val * 4, 0.1); \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), Val * 4);        \
    }                                              \
  }
TEST_LLVM_SCALAR_TYPES(MUL_TEST)
#undef MUL_TEST

#define DIV_TEST(Type, Name, Val)                  \
  TEST(LLVM, Name##DivTest) {                      \
    KernelScope kernel_scope;                      \
    auto a = Name##Imm::make((Type)6);             \
    auto b = Name##Imm::make((Type)3);             \
    auto c = Div::make(a, b);                      \
    LLVMExprEval cg(c);                            \
    if (std::is_floating_point<decltype(Val)>()) { \
      ASSERT_NEAR(cg.value<Type>(), 2, 0.1);       \
    } else {                                       \
      ASSERT_EQ(cg.value<Type>(), 2);              \
    }                                              \
  }
TEST_LLVM_SCALAR_TYPES(DIV_TEST)
#undef DIV_TEST

TEST(LLVM, IntToFloatCastTest) {
  KernelScope kernel_scope;
  auto a = IntImm::make(2);
  auto b = Cast::make(kFloat, a);
  LLVMExprEval cg(b, {});
  ASSERT_EQ(cg.value<float>(), 2.0);
}

TEST(LLVM, FloatToIntCastTest) {
  KernelScope kernel_scope;
  auto a = FloatImm::make(2.0);
  auto b = Cast::make(kInt, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int>(), 2);
}

TEST(LLVM, IntToLongCastTest) {
  KernelScope kernel_scope;
  auto a = IntImm::make(12345);
  auto b = Cast::make(kLong, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int64_t>(), 12345);
}

TEST(LLVM, ByteToCharCastTest) {
  KernelScope kernel_scope;
  auto a = ByteImm::make(250);
  auto b = Cast::make(kChar, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int8_t>(), (int8_t)250);
}

TEST(LLVM, HalfToLongCastTest) {
  KernelScope kernel_scope;
  auto a = HalfImm::make(2.0);
  auto b = Cast::make(kLong, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int64_t>(), 2);
}

TEST(LLVM, ByteToDoubleCastTest) {
  KernelScope kernel_scope;
  auto a = ByteImm::make(2);
  auto b = Cast::make(kDouble, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<double>(), 2);
}

TEST(LLVM, BitCast) {
  constexpr int16_t ref16 = 1337;
  constexpr int32_t ref32 = 1337;
  constexpr int64_t ref64 = 1337;
  at::Half reff16 = 1337.0f;
  constexpr float reff32 = 1337.0f;
  constexpr double reff64 = 1337.0f;

  // this is broken
  /*{
    KernelScope kernel_scope;
    at::Half k_;
    at::Half* k = &k_;
    *reinterpret_cast<int16_t*>(k) = ref16;
    auto a = HalfImm::make(k);
    auto b = BitCast::make(kShort, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<int16_t>(), ref16);
  }*/

  {
    KernelScope kernel_scope;
    float k = raw_bitcast<float>(ref32);
    auto a = FloatImm::make(k);
    auto b = BitCast::make(kInt, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<int32_t>(), ref32);
  }

  {
    KernelScope kernel_scope;
    double k = raw_bitcast<double>(ref64);
    auto a = DoubleImm::make(k);
    auto b = BitCast::make(kLong, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<int64_t>(), ref64);
  }

  {
    KernelScope kernel_scope;
    int64_t k = raw_bitcast<int64_t>(reff64);
    auto a = LongImm::make(k);
    auto b = BitCast::make(kDouble, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<double>(), reff64);
  }

  {
    KernelScope kernel_scope;
    int32_t k = raw_bitcast<int32_t>(reff32);
    auto a = IntImm::make(k);
    auto b = BitCast::make(kFloat, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<float>(), reff32);
  }
}

TEST(LLVM, fastLogFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128 * 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, fast_log(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  LLVMCodeGen ir_eval(stmt, {a_buf, b_buf});
  ir_eval.call({a_v, b_v});

  for (int i = 0; i < kTotalSize; ++i) {
    auto test = b_v(i);
    auto ref = std::log(a_v(i));
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      ASSERT_FLOAT_EQ(test, ref);
    }
  }
}

TEST(LLVM, LetTest01) {
  KernelScope kernel_scope;

  Placeholder a(BufHandle("A", {1}, kFloat));
  std::vector<float> v = {1, 0};
  std::vector<void*> args({v.data()});
  VarHandle x("x", kFloat);
  auto block = Block::make({
      Let::make(x, 3.f),
      a.store({0}, ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f))),
  });

  LLVMCodeGen cg(block, {a});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(v[0], 2.f + 3.f * 3.f + 4.f);
}

TEST(LLVM, LetTest02) {
  KernelScope kernel_scope;

  Placeholder a(BufHandle("A", {1}, kFloat));
  std::vector<float> v = {1, 0};
  std::vector<void*> args({v.data()});
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  auto block = Block::make(
      {Let::make(x, 3.f),
       Let::make(y, 6.f),
       a.store(
           {IntImm::make(0)},
           ExprHandle(2.f) + (x * ExprHandle(3.f) + y * ExprHandle(4.f)))});

  LLVMCodeGen cg(block, {a});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(v[0], 2.f + 3.f * 3.f + 6.f * 4.f);
}

TEST(LLVM, LetTestMultitype) {
  KernelScope kernel_scope;

  Placeholder a(BufHandle("A", {1}, kDouble));
  std::vector<double> v = {1, 0};
  std::vector<void*> args({v.data()});
  VarHandle x("x", kByte);
  VarHandle y("y", kHalf);
  auto block = Block::make(
      {Let::make(x, 3),
       Let::make(y, 6.f),
       a.store(
           {0},
           Cast::make(
               kDouble,
               ExprHandle(2.f) +
                   (x * ExprHandle(3.f) + y * ExprHandle(4.f))))});

  LLVMCodeGen cg(block, {a});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(v[0], 2.f + 3 * 3.f + 6.f * 4.f);
}

TEST(LLVM, BufferTest) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {32}, kFloat));
  std::vector<int32_t> v(5);
  std::vector<void*> args({v.data()});
  auto rv = IntImm::make(0);
  LLVMExprEval cg(rv, {a});
  ASSERT_EQ(cg.value<int>(args), 0);
}

TEST(LLVM, BlockTest) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {32}, kInt));
  std::vector<int32_t> v = {1, 2};
  std::vector<void*> args({v.data()});

  auto block = Block::make({
      a.store({0}, 3),
      a.store({1}, 4),
      a.store({0}, 4),
  });

  LLVMCodeGen cg(block, {a});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(v[0], 4);
  ASSERT_EQ(v[1], 4);
}

TEST(LLVM, LoadStoreTest) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {1}, kInt));
  Placeholder b(BufHandle("B", {1}, kInt));
  std::vector<int32_t> a_buffer = {42};
  std::vector<int32_t> b_buffer = {-11};

  auto store = b.store({0}, a.load(0));
  LLVMCodeGen cg(store, {a, b});
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(a_buffer[0], 42);
  ASSERT_EQ(b_buffer[0], 42);
}

TEST(LLVM, IfThenElseTest) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {1}, kInt));
  Placeholder b(BufHandle("B", {1}, kInt));
  Placeholder c(BufHandle("C", {1}, kInt));
  std::vector<int32_t> a_buffer = {42};
  std::vector<int32_t> b_buffer = {-11};
  std::vector<int32_t> c_buffer = {1};

  auto store = b.store({0}, IfThenElse::make(c.load(0), a.load(0), 0));
  LLVMCodeGen cg(store, {a, b, c});
  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(a_buffer[0], 42);
  ASSERT_EQ(b_buffer[0], 42);
}

// if (x < 10) x = x + 1
TEST(LLVM, CondNoFalseBlockTest) {
  KernelScope kernel_scope;

  Placeholder x(BufHandle("X", {1}, kInt));
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  auto cond = Cond::make(cmp, x.store({0}, x.load(0) + 1), nullptr);

  for (int32_t x_value : {0, 10, 20}) {
    std::vector<int32_t> x_buffer = {x_value};
    std::vector<void*> args({x_buffer.data()});
    LLVMCodeGen cg(cond, {x});
    ASSERT_EQ(cg.value<int>(args), 0);
    if (x_value < 10) {
      ASSERT_EQ(x_buffer[0], x_value + 1);
    } else {
      ASSERT_EQ(x_buffer[0], x_value);
    }
  }
}

// if (x < 10) {
//   x = x + 1;
// } else {
//   x = x - 1;
// }
TEST(LLVM, CondTest) {
  KernelScope kernel_scope;

  Placeholder x(BufHandle("X", {1}, kInt));
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  auto cond =
      Cond::make(cmp, x.store({0}, x.load(0) + 1), x.store({0}, x.load(0) - 1));
  auto block = Block::make({
      cond,
      x.store({0}, x.load(0) * 2),
  });

  for (int32_t x_value : {0, 10, 20}) {
    std::vector<int32_t> x_buffer = {x_value};
    std::vector<void*> args({x_buffer.data()});
    LLVMCodeGen cg(block, {x});
    ASSERT_EQ(cg.value<int>(args), 0);
    if (x_value < 10) {
      ASSERT_EQ(x_buffer[0], (x_value + 1) * 2);
    } else {
      ASSERT_EQ(x_buffer[0], (x_value - 1) * 2);
    }
  }
}

// if (x < 10) {
//   if (x > 5) {
//     x = x + 1;
//   } else {
//     x = x - 1;
//   }
// } else {
//   if (x <= 15) {
//     x = x + 2;
//   } else {
//     x = x - 2;
//   }
// }
TEST(LLVM, CondNestedTest) {
  KernelScope kernel_scope;

  Placeholder x(BufHandle("X", {1}, kInt));
  auto true_cmp =
      CompareSelect::make(x.load(0), 5, CompareSelectOperation::kGT);
  auto true_cond = Cond::make(
      true_cmp, x.store({0}, x.load(0) + 1), x.store({0}, x.load(0) - 1));
  auto false_cmp =
      CompareSelect::make(x.load(0), 15, CompareSelectOperation::kLE);
  auto false_cond = Cond::make(
      false_cmp, x.store({0}, x.load(0) + 2), x.store({0}, x.load(0) - 2));
  auto cmp = CompareSelect::make(x.load(0), 10, CompareSelectOperation::kLT);
  auto cond = Cond::make(cmp, true_cond, false_cond);

  for (int32_t x_value : {0, 8, 15, 20}) {
    std::vector<int32_t> x_buffer = {x_value};
    std::vector<void*> args({x_buffer.data()});
    LLVMCodeGen cg(cond, {x});
    ASSERT_EQ(cg.value<int>(args), 0);
    if (x_value < 10) {
      if (x_value > 5) {
        ASSERT_EQ(x_buffer[0], x_value + 1);
      } else {
        ASSERT_EQ(x_buffer[0], x_value - 1);
      }
    } else {
      if (x_value <= 15) {
        ASSERT_EQ(x_buffer[0], x_value + 2);
      } else {
        ASSERT_EQ(x_buffer[0], x_value - 2);
      }
    }
  }
}

TEST(LLVM, DirectVectorization) {
  KernelScope ks;
  constexpr int M = 3;
  constexpr int N = 64;
  BufHandle a("a", {M, N}, kFloat);
  BufHandle b("b", {M, N}, kFloat);
  BufHandle c("c", {M, N}, kFloat);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  Stmt* s = For::make(
      m,
      0,
      M,
      Store::make(
          c,
          {Ramp::make(m * 64, 1, 64)},
          Load::make(
              {kFloat, 64},
              a,
              {Ramp::make(m * 64, 1, 64)},
              Broadcast::make(1, 64)) *
              Load::make(
                  {kFloat, 64},
                  b,
                  {Ramp::make(m * 64, 1, 64)},
                  Broadcast::make(1, 64)),
          Broadcast::make(1, 64)));
  LLVMCodeGen cg(s, {a, b, c});
}

TEST(LLVM, VecLoadStoreTest) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {1}, kInt));
  Placeholder b(BufHandle("B", {1}, kInt));
  std::vector<int32_t> a_buffer = {1, 1, 1, 1};
  std::vector<int32_t> b_buffer = {2, 2, 2, 2};

  auto store = b.storeWithMask(
      {Ramp::make(0, 1, 4)},
      a.loadWithMask(
          {Ramp::make(0, 1, 4)}, Broadcast::make(IntImm::make(1), 4)),
      Broadcast::make(IntImm::make(1), 4));
  LLVMCodeGen cg(store, {a, b});
  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(a_buffer[0], 1);
  ASSERT_EQ(a_buffer[1], 1);
  ASSERT_EQ(a_buffer[2], 1);
  ASSERT_EQ(a_buffer[3], 1);
  ASSERT_EQ(b_buffer[0], 1);
  ASSERT_EQ(b_buffer[1], 1);
  ASSERT_EQ(b_buffer[2], 1);
  ASSERT_EQ(b_buffer[3], 1);
}

#define FLOAT_INTRINSICS_TEST(Name, Lanes)                       \
  TEST(LLVM, VecFloat_##Name##Lane##Lanes##Test) {               \
    KernelScope kernel_scope;                                    \
    Placeholder a(BufHandle("A", {1}, kFloat));                  \
    Placeholder b(BufHandle("B", {1}, kFloat));                  \
    float val = 0.5f;                                            \
    std::vector<float> a_buffer(Lanes, val);                     \
    std::vector<float> b_buffer(Lanes, val);                     \
    auto store = b.storeWithMask(                                \
        {Ramp::make(0, 1, Lanes)},                               \
        Name(a.loadWithMask(                                     \
            {Ramp::make(0, 1, Lanes)},                           \
            Broadcast::make(IntImm::make(1), Lanes))),           \
        Broadcast::make(IntImm::make(1), Lanes));                \
    LLVMCodeGen cg(store, {a, b});                               \
    std::vector<void*> args({a_buffer.data(), b_buffer.data()}); \
    ASSERT_EQ(cg.value<int>(args), 0);                           \
    for (int i = 0; i < Lanes; i++) {                            \
      ASSERT_FLOAT_EQ(a_buffer[i], val);                         \
    }                                                            \
  } // namespace jit
FLOAT_INTRINSICS_TEST(erf, 4)
FLOAT_INTRINSICS_TEST(erfc, 4)
FLOAT_INTRINSICS_TEST(acos, 4)
FLOAT_INTRINSICS_TEST(asin, 4)
FLOAT_INTRINSICS_TEST(atan, 4)
FLOAT_INTRINSICS_TEST(cosh, 4)
FLOAT_INTRINSICS_TEST(sinh, 4)
FLOAT_INTRINSICS_TEST(tanh, 4)
FLOAT_INTRINSICS_TEST(expm1, 4)
FLOAT_INTRINSICS_TEST(lgamma, 4)
FLOAT_INTRINSICS_TEST(erf, 8)
FLOAT_INTRINSICS_TEST(erfc, 8)
FLOAT_INTRINSICS_TEST(acos, 8)
FLOAT_INTRINSICS_TEST(asin, 8)
FLOAT_INTRINSICS_TEST(atan, 8)
FLOAT_INTRINSICS_TEST(cosh, 8)
FLOAT_INTRINSICS_TEST(sinh, 8)
FLOAT_INTRINSICS_TEST(tanh, 8)
FLOAT_INTRINSICS_TEST(expm1, 8)
FLOAT_INTRINSICS_TEST(lgamma, 8)
#undef FLOAT_INTRINSICS_TEST

#define DOUBLE_INTRINSICS_TEST(Name, Lanes)                      \
  TEST(LLVM, VecDouble_##Name##Lane##Lanes##Test) {              \
    KernelScope kernel_scope;                                    \
    Placeholder a(BufHandle("A", {1}, kDouble));                 \
    Placeholder b(BufHandle("B", {1}, kDouble));                 \
    float val = 0.5f;                                            \
    std::vector<double> a_buffer(Lanes, val);                    \
    std::vector<double> b_buffer(Lanes, val);                    \
    auto store = b.storeWithMask(                                \
        {Ramp::make(0, 1, Lanes)},                               \
        Name(a.loadWithMask(                                     \
            {Ramp::make(0, 1, Lanes)},                           \
            Broadcast::make(IntImm::make(1), Lanes))),           \
        Broadcast::make(IntImm::make(1), Lanes));                \
    LLVMCodeGen cg(store, {a, b});                               \
    std::vector<void*> args({a_buffer.data(), b_buffer.data()}); \
    ASSERT_EQ(cg.value<int>(args), 0);                           \
    for (int i = 0; i < Lanes; i++) {                            \
      ASSERT_FLOAT_EQ(a_buffer[i], val);                         \
    }                                                            \
  } // namespace jit
DOUBLE_INTRINSICS_TEST(erf, 2)
DOUBLE_INTRINSICS_TEST(erfc, 2)
DOUBLE_INTRINSICS_TEST(acos, 2)
DOUBLE_INTRINSICS_TEST(asin, 2)
DOUBLE_INTRINSICS_TEST(atan, 2)
DOUBLE_INTRINSICS_TEST(cosh, 2)
DOUBLE_INTRINSICS_TEST(sinh, 2)
DOUBLE_INTRINSICS_TEST(tanh, 2)
DOUBLE_INTRINSICS_TEST(expm1, 2)
DOUBLE_INTRINSICS_TEST(lgamma, 2)
DOUBLE_INTRINSICS_TEST(erf, 4)
DOUBLE_INTRINSICS_TEST(erfc, 4)
DOUBLE_INTRINSICS_TEST(acos, 4)
DOUBLE_INTRINSICS_TEST(asin, 4)
DOUBLE_INTRINSICS_TEST(atan, 4)
DOUBLE_INTRINSICS_TEST(cosh, 4)
DOUBLE_INTRINSICS_TEST(sinh, 4)
DOUBLE_INTRINSICS_TEST(tanh, 4)
DOUBLE_INTRINSICS_TEST(expm1, 4)
DOUBLE_INTRINSICS_TEST(lgamma, 4)
#undef DOUBLE_INTRINSICS_TEST

TEST(LLVM, VectorizerLoadStoreTest) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {1}, kInt));

  Tensor* c =
      Compute("c", {{4, "i"}}, [&](const VarHandle& i) { return a.load(i); });

  Placeholder c_buf(BufHandle(c->buf()));
  LoopNest l({c});
  Stmt* s = l.root_stmt();
  l.vectorize(dynamic_cast<For*>(dynamic_cast<Block*>(s)->front()));

  ASSERT_TRUE(dynamic_cast<For*>(dynamic_cast<Block*>(s)->front()) == nullptr);

  LLVMCodeGen cg(s, {a, c_buf});

  std::vector<int> a_vec(4, 21);
  std::vector<int> c_vec(4, 0);
  std::vector<void*> args({a_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 21);
}

TEST(LLVM, VectorizeBitCast) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("A", {128}, kInt));

  Tensor* c = Compute("c", {{128, "i"}}, [&](const VarHandle& i) {
    return bitcast<float>(a.load(i));
  });

  Placeholder c_buf(BufHandle(c->buf()));
  LoopNest l({c});
  Stmt* s = l.root_stmt();
  l.vectorize(dynamic_cast<For*>(dynamic_cast<Block*>(s)->front()));
  ASSERT_TRUE(dynamic_cast<For*>(dynamic_cast<Block*>(s)->front()) == nullptr);

  LLVMCodeGen cg(s, {a, c_buf});

  std::vector<int> a_vec(128);
  std::vector<float> c_vec(128);
  for (auto i = 0; i < 128; ++i) {
    a_vec[i] = raw_bitcast<int>(1337.f);
  }
  std::vector<void*> args({a_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 1337.f);
}

TEST(LLVM, MemcpyTest) {
  KernelScope kernel_scope;
  constexpr int N = 32;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  std::vector<int32_t> a_buffer(N, 42);
  std::vector<int32_t> b_buffer(N, 0);

  VarHandle i("i", kInt);
  auto expr = For::make(i, 0, N, b.store({i}, a.load(i)));

  LLVMCodeGen cg(expr, {a, b});

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, 42);
  assertAllEqual(b_buffer, 42);
}

TEST(LLVM, BzeroTest) {
  KernelScope kernel_scope;
  constexpr int N = 32;
  Placeholder b(BufHandle("B", {N}, kInt));
  std::vector<int32_t> b_buffer(N, 11);

  VarHandle i("i", kInt);
  auto expr = For::make(i, 0, N, b.store({i}, 0));

  LLVMCodeGen cg(expr, {b});

  std::vector<void*> args({b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(b_buffer, 0);
}

TEST(LLVM, ElemwiseAdd) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int32_t> a_buffer(N, 41);
  std::vector<int32_t> b_buffer(N, 1);
  std::vector<int32_t> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr = For::make(i, 0, N, c.store({i}, Add::make(a.load(i), b.load(i))));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 42);
}

TEST(LLVM, ElemwiseAddFloat) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  Placeholder c(BufHandle("C", {N}, kFloat));
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr = For::make(i, 0, N, c.store({i}, a.load(i) + b.load(i)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 42.0f);
}

TEST(LLVM, ElemwiseLog10Float) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  std::vector<float> a_buffer(N, 10.0f);
  std::vector<float> b_buffer(N, 2.0f);

  auto mask = Broadcast::make(IntImm::make(1), 4);
  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N / 4,
      b.storeWithMask(
          {Ramp::make(i * 4, 1, 4)},
          log10(a.loadWithMask({Ramp::make(i * 4, 1, 4)}, mask)),
          mask));

  LLVMCodeGen cg(expr, {a, b});

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, 10.0f);
  assertAllEqual(b_buffer, 1.0f);
}

TEST(LLVM, ElemwiseLog1pFloat) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  std::vector<float> a_buffer(N, expf(3.0f) - 1);
  std::vector<float> b_buffer(N, 42.0f);

  auto mask = Broadcast::make(IntImm::make(1), 4);
  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N / 4,
      b.storeWithMask(
          {Ramp::make(i * 4, 1, 4)},
          log1p(a.loadWithMask({Ramp::make(i * 4, 1, 4)}, mask)),
          mask));

  LLVMCodeGen cg(expr, {a, b});

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, expf(3.0f) - 1);
  ExpectAllNear(b_buffer, 3.0f, 1e-5f);
}

TEST(LLVM, ElemwiseMaxInt) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr =
      For::make(i, 0, N, c.store({i}, Max::make(a.load(i), b.load(i), false)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 41);
}

TEST(LLVM, ElemwiseMinInt) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 41);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr =
      For::make(i, 0, N, c.store({i}, Min::make(a.load(i), b.load(i), false)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 1);
}

TEST(LLVM, ElemwiseMaxFloat) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  Placeholder c(BufHandle("C", {N}, kFloat));
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr =
      For::make(i, 0, N, c.store({i}, Max::make(a.load(i), b.load(i), false)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 41.0f);
}

TEST(LLVM, ElemwiseMaxNaNFloat) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  Placeholder c(BufHandle("C", {N}, kFloat));
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr =
      For::make(i, 0, N, c.store({i}, Max::make(a.load(i), b.load(i), false)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(b_buffer, 1.0f);
  for (auto const& elt : c_buffer) {
    ASSERT_TRUE(std::isnan(elt));
  }
}

TEST(LLVM, ElemwiseMinFloat) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  Placeholder c(BufHandle("C", {N}, kFloat));
  std::vector<float> a_buffer(N, 41);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr =
      For::make(i, 0, N, c.store({i}, Min::make(a.load(i), b.load(i), false)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1.0f);
}

TEST(LLVM, ElemwiseMinNaNFloat) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  Placeholder c(BufHandle("C", {N}, kFloat));
  std::vector<float> a_buffer(N, NAN);
  std::vector<float> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto expr =
      For::make(i, 0, N, c.store({i}, Min::make(a.load(i), b.load(i), false)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(b_buffer, 1.0f);
  for (auto const& elt : c_buffer) {
    ASSERT_TRUE(std::isnan(elt));
  }
}

TEST(LLVM, ElemwiseMod) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int32_t> a_buffer(N, 41);
  std::vector<int32_t> b_buffer(N, 23);
  std::vector<int32_t> c_buffer(N, 18);

  VarHandle i("i", kInt);
  auto expr = For::make(i, 0, N, c.store({i}, Mod::make(a.load(i), b.load(i))));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);
  assertAllEqual(a_buffer, 41);
  assertAllEqual(b_buffer, 23);
  assertAllEqual(c_buffer, 18);
}

TEST(LLVM, CompareSelectIntEQ) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  for (int i = 0; i < N / 2; i++) {
    b_buffer[i] = 0;
    c_ref[i] = 0;
  }

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectFloatEQ) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kFloat));
  Placeholder b(BufHandle("B", {N}, kFloat));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<float> a_buffer(N, 1.0f);
  std::vector<float> b_buffer(N, 1.0f);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1.0f);
  assertAllEqual(b_buffer, 1.0f);
  assertAllEqual(c_buffer, 1);
}

TEST(LLVM, CompareSelectByteGT) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kByte));
  Placeholder b(BufHandle("B", {N}, kByte));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 0);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 0);

  for (int i = 0; i < N / 2; i++) {
    a_buffer[i] = 128;
    c_ref[i] = 1;
  }

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGT)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(b_buffer, uint8_t(0));
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectByteGE) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kByte));
  Placeholder b(BufHandle("B", {N}, kByte));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 0);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGE)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(b_buffer, uint8_t(0));
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectByteLT) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kByte));
  Placeholder b(BufHandle("B", {N}, kByte));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 128);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  for (int i = 0; i < N / 2; i++) {
    a_buffer[i] = 128;
    c_ref[i] = 0;
  }

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLT)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(b_buffer, uint8_t(128));
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectByteLE) {
  KernelScope kernel_scope;
  constexpr int N = 1024;
  Placeholder a(BufHandle("A", {N}, kByte));
  Placeholder b(BufHandle("B", {N}, kByte));
  Placeholder c(BufHandle("C", {N}, kInt));
  std::vector<uint8_t> a_buffer(N, 0);
  std::vector<uint8_t> b_buffer(N, 128);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 1);

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLE)));

  LLVMCodeGen cg(expr, {a, b, c});

  std::vector<void*> args({a_buffer.data(), b_buffer.data(), c_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(b_buffer, uint8_t(128));
  for (int i = 0; i < N; i++) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, StoreFloat) {
  KernelScope kernel_scope;
  Placeholder result(BufHandle("result", {1}, kFloat));
  std::vector<float> result_buffer = {0.0f};
  auto expr = result.store({0}, FloatImm::make(3.14f));
  LLVMCodeGen cg(expr, {result});
  std::vector<void*> args({result_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(result_buffer[0], 3.14f);
}

TEST(LLVM, SimpleMath01) {
  KernelScope kernel_scope;
  const int N = 1024;
  Tensor* tensor = Compute("f", {{N, "i"}}, [](const VarHandle& i) {
    return cast<float>(i * i + 1);
  });
  LoopNest l({tensor});
  Stmt* stmt = l.root_stmt();
  Placeholder f_buf(BufHandle(tensor->buf()));
  LLVMCodeGen cg(stmt, {f_buf});

  PaddedBuffer<float> f_v(N, "f_v");
  std::vector<void*> args({f_v.data()});
  int value = cg.value<int>(args);
  ASSERT_EQ(value, 0);
  PaddedBuffer<float> f_ref(N, "f_ref");
  for (int i = 0; i < N; i++) {
    f_ref(i) = i * i + 1;
  }
  ExpectAllNear(f_v, f_ref, 1e-5);
}

TEST(LLVM, ComputeMul) {
  KernelScope kernel_scope;
  const int N = 1024;
  Placeholder a(BufHandle("a", {N}, kFloat));
  Placeholder b(BufHandle("b", {N}, kFloat));
  Tensor* c = Compute("c", {{N, "i"}}, [&](const VarHandle& i) {
    return a.load(i) * b.load(i);
  });

  Placeholder c_buf(BufHandle(c->buf()));
  LoopNest l({c});
  Stmt* s = l.root_stmt();

  LLVMCodeGen cg(s, {a, b, c_buf});

  std::vector<float> a_vec(N, 21.0f);
  std::vector<float> b_vec(N, 2.0f);
  std::vector<float> c_vec(N, 0.0f);
  std::vector<void*> args({a_vec.data(), b_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 42.0f);
}

TEST(LLVM, BroadcastAdd) {
  KernelScope kernel_scope;
  const int M = 32;
  const int N = 1024;
  Placeholder a(BufHandle("a", {M, N}, kFloat));
  Placeholder b(BufHandle("b", {N}, kFloat));
  Tensor* c = Compute(
      "c", {{M, "i"}, {N, "j"}}, [&](const VarHandle& i, const VarHandle& j) {
        return a.load(i, j) + b.load(j);
      });

  Placeholder c_buf(BufHandle(c->buf()));
  LoopNest l({c});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();

  LLVMCodeGen cg(s, {a, b, c_buf});

  std::vector<float> av(M * N);
  std::iota(av.begin(), av.end(), 0);
  std::vector<float> bv(N);
  std::iota(bv.begin(), bv.end(), 0);
  std::vector<float> cv(M * N, 0);
  std::vector<void*> args({av.data(), bv.data(), cv.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      ASSERT_EQ(cv[i * N + j], av[i * N + j] + bv[j]);
    }
  }
}

TEST(LLVM, BitwiseOps) {
  KernelScope kernel_scope;
  auto a = IntImm::make(59);
  auto b = IntImm::make(11);
  auto c = IntImm::make(101);
  auto d = IntImm::make(2);

  ExprHandle f = (((a ^ (b << 1)) & c) >> 2) | d;
  LLVMExprEval cg(f);

  ASSERT_EQ(cg.value<int>(), 11);
}

TEST(LLVM, ArithmeticRightShift) {
  KernelScope ks;
  auto a = CharImm::make(-4);
  auto b = CharImm::make(1);
  ExprHandle f = a >> b;
  LLVMExprEval cg(f);
  ASSERT_EQ(cg.value<int8_t>(), -2);
}

TEST(LLVM, LogicalRightShift) {
  KernelScope ks;
  auto a = ByteImm::make(0xfc);
  auto b = ByteImm::make(1);
  ExprHandle f = a >> b;
  LLVMExprEval cg(f);
  ASSERT_EQ(cg.value<uint8_t>(), 0x7e);
}

TEST(LLVM, DynamicShapeAdd) {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);
    Placeholder a(BufHandle("a", {n}, kFloat));
    Placeholder b(BufHandle("b", {n}, kFloat));
    Placeholder c(BufHandle("c", {n}, kFloat));
    VarHandle i("i", kInt);
    Stmt* s = For::make(i, 0, n, c.store({i}, a.load(i) + b.load(i)));
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    LLVMCodeGen cg(s, {a, b, c, n});
    std::vector<void*> args({aData.data(), bData.data(), cData.data(), &size});
    cg.value<float>(args);
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

TEST(LLVM, BindDynamicShapeAdd) {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);
    Placeholder a(BufHandle("a", {n}, kFloat));
    Placeholder b(BufHandle("b", {n}, kFloat));
    Placeholder c(BufHandle("c", {n}, kFloat));
    VarHandle i("i", kInt);
    Stmt* s = For::make(i, 0, n, c.store({i}, a.load(i) + b.load(i)));
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    LLVMCodeGen cg(s, {a, b, c, n});
    cg.call({aData, bData, cData, size});
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

TEST(LLVM, TensorDynamicShapeAdd) {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);
    Placeholder a(BufHandle("a", {n}, kFloat));
    Placeholder b(BufHandle("b", {n}, kFloat));
    Tensor* c = Compute("c", {{n, "n"}}, [&](const VarHandle& i) {
      return a.load(i) + b.load(i);
    });
    LoopNest l({c});
    Stmt* s = l.root_stmt();
    LLVMCodeGen cg(s, {a, b, c, n});
    std::vector<float> aData(size, 1.0f);
    std::vector<float> bData(size, 2.0f);
    std::vector<float> cData(size, 0.0f);
    cg.call({aData, bData, cData, size});
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

TEST(LLVM, DynamicShape2D) {
  KernelScope kernel_scope;
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    Placeholder a(BufHandle("a", {m, n}, kFloat));
    Placeholder b(BufHandle("b", {m, n}, kFloat));
    Tensor* c = Compute(
        "c", {{m, "m"}, {n, "n"}}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    LoopNest l({c});
    l.prepareForCodegen();
    Stmt* s = l.root_stmt();
    LLVMCodeGen cg(s, {a, b, c, m, n});
    std::vector<float> aData(M * N, 1.0f);
    std::vector<float> bData(M * N, 2.0f);
    std::vector<float> cData(M * N, 0.0f);
    cg.call({aData, bData, cData, M, N});
    ExpectAllNear(cData, std::vector<float>(M * N, 3.0f), 1e-7);
  };
  testWithSize(1, 8);
  testWithSize(16, 32);
  testWithSize(37, 11);
}

TEST(LLVM, EmptyStmt) {
  KernelScope kernel_scope;
  Stmt* s = new Block({});

  LLVMCodeGen cg(s, {});
  cg.call({});
  // Just don't crash.
}

TEST(LLVM, EliminatedStmt) {
  KernelScope kernel_scope;
  Placeholder a(BufHandle("a", {1}, kFloat));

  Tensor* c = Compute("c", {{0, "m"}}, [&](const VarHandle& m) { return m; });

  LoopNest l({c});
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();
  s = IRSimplifier::simplify(s);
  LLVMCodeGen cg(s, {a, c});
  std::vector<float> aData(1, 1.0f);
  std::vector<float> cData(0, 0.0f);
  cg.call({aData, cData});
}

TEST(LLVM, SimpleReduction) {
  KernelScope kernel_scope;

  int M = 128;
  int N = 64;
  const int kTotalSize = M * N;

  Placeholder a("a", kFloat, {1, M, N});

  // TODO: why doesn't implicit vector<DimArg> work?
  std::vector<DimArg> axis = {DimArg(1)};
  std::vector<DimArg> reduce_axis = {DimArg(M), DimArg(N)};
  Tensor* b = Reduce("sum", axis, Sum(), a, reduce_axis);
  LoopNest loop({b});

  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  LLVMCodeGen cg(s, {a, b});

  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  cg.call({a_v, b_v});

  ExpectAllNear(b_v, b_ref, 1e-5);
}

TEST(LLVM, RFactorReduction) {
  KernelScope kernel_scope;

  int M = 128;
  int N = 64;
  const int kTotalSize = M * N;

  Placeholder a("a", kFloat, {1, M, N});

  // TODO: why doesn't implicit vector<DimArg> work?
  std::vector<DimArg> axis = {DimArg(1)};
  std::vector<DimArg> reduce_axis = {DimArg(M), DimArg(N)};
  Tensor* b = Reduce("sum", axis, Sum(), a, reduce_axis);
  LoopNest loop({b});

  std::vector<For*> loops = loop.getLoopStmtsFor(b);
  For* loop_m = loops.at(1);
  For* loop_n = loops.at(2);
  loop.reorderAxis(loop_m, loop_n);

  loops = loop.getLoopStmtsFor(b);
  loop_m = loops.at(2);
  loop_n = loops.at(1);
  auto b_body = NodeFinder<ReduceOp>::find(loop.root_stmt())[0];
  loop.rfactor(b_body, loop_n->var(), loop_n->body());

  loop.prepareForCodegen();
  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  LLVMCodeGen cg(s, {a, b});

  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  cg.call({a_v, b_v});

  ExpectAllNear(b_v, b_ref, 1e-5);
}

TEST(LLVM, RFactorVectorizedReduction) {
  KernelScope kernel_scope;

  int M = 128;
  int N = 64;
  const int kTotalSize = M * N;

  Placeholder a("a", kFloat, {1, M, N});

  Tensor* b = Reduce("sum", {{1, "K"}}, Sum(), a, {{M, "M"}, {N, "N"}});
  LoopNest loopnest({b});
  std::vector<For*> loops = loopnest.getLoopStmtsFor(b);
  For* loop_k = loops.at(0);
  For* loop_m = loops.at(1);
  For* loop_n = loops.at(2);
  auto b_body = NodeFinder<ReduceOp>::find(loopnest.root_stmt())[0];
  loopnest.rfactor(b_body, loop_n->var());

  loops = NodeFinder<For>::find(loopnest.root_stmt());
  loop_k = loops.at(0);
  // loop 1 is the initializer of tmp_buf
  loop_m = loops.at(2);
  loop_n = loops.at(3);
  loopnest.reorderAxis(loop_n, loop_m);

  // Case-III reductions
  loops = NodeFinder<For>::find(loopnest.root_stmt());
  // Vectorize initializer of tmp_buf
  loopnest.vectorize(loops[1]);
  // Vectorize producer of tmp_buf
  loopnest.vectorize(loops[2]);

  loopnest.prepareForCodegen();

  Stmt* s = IRSimplifier::simplify(loopnest.root_stmt());
  LLVMCodeGen cg(s, {a, b});

  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  cg.call({a_v, b_v});

  ExpectAllNear(b_v, b_ref, 1e-5);
}

TEST(LLVM, SimpleParallel) {
  for (int test_cfg = 0; test_cfg < 4; test_cfg++) {
    // Compute a simple operation, and try all loop-axis combination to be
    // parallel or sequential.
    ExecutionCounter counter(llvm_codegen_parallel_dispatched);
    KernelScope kernel_scope;
    const int M = 4;
    const int N = 6;
    Tensor* f = Compute(
        "f", {{M, "m"}, {N, "n"}}, [](const VarHandle& m, const VarHandle& n) {
          return cast<float>(m + n);
        });
    LoopNest loop_nest({f});
    auto const& loops = loop_nest.getLoopStmtsFor(f);
    For* m = loops[0];
    For* n = loops[1];
    if (test_cfg & 0x1) {
      m->set_parallel();
    }
    if (test_cfg & 0x2) {
      n->set_parallel();
    }
    loop_nest.prepareForCodegen();
    Stmt* stmt = loop_nest.root_stmt();
    LLVMCodeGen cg(stmt, {f});

    PaddedBuffer<float> f_v(M, N, "f_v");
    std::vector<void*> args({f_v.data()});
    int value = cg.value<int>(args);
    ASSERT_EQ(value, 0);
    PaddedBuffer<float> f_ref(M, N, "f_ref");
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        f_ref(m, n) = m + n;
      }
    }
    ExpectAllNear(f_v, f_ref, 1e-5);
    int count = counter.elapsed_value();
    if (test_cfg == 0) {
      ASSERT_EQ(count, 0);
    } else {
      ASSERT_GT(count, 0);
    }
  }
}

TEST(LLVM, CompositeParallel) {
  int loop_count = 6;
  int test_count = 1 << loop_count;
  // Compute a composite operation, and try all loop-axis combination to be
  // parallel or sequential.
  for (int test_cfg = 0; test_cfg < test_count; test_cfg++) {
    ExecutionCounter counter(llvm_codegen_parallel_dispatched);
    KernelScope kernel_scope;
    int M = 5;
    int N = 7;
    Tensor* t1 =
        Compute("t1", {{M, "M"}}, [](const VarHandle& m) { return m + 1.f; });
    Tensor* t2 =
        Compute("t2", {{N, "N"}}, [](const VarHandle& n) { return n + 2.f; });
    Tensor* t3 = Compute(
        "t3",
        {{M, "M"}, {N, "N"}},
        [=](const VarHandle& m, const VarHandle& n) {
          return t1->call(m) * t2->call(n);
        });
    Tensor* t4 = Compute(
        "t4",
        {{M, "M"}, {N, "N"}},
        [=](const VarHandle& m, const VarHandle& n) {
          return t3->call(m, n) + m + n;
        });
    LoopNest loop_nest({t4}, {t1, t2, t3, t4});
    std::vector<For*> loop_list;
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t1);
      loop_list.push_back(loops[0]);
    }
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t2);
      loop_list.push_back(loops[0]);
    }
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t3);
      loop_list.push_back(loops[0]);
      loop_list.push_back(loops[1]);
    }
    {
      auto const& loops = loop_nest.getLoopStmtsFor(t4);
      loop_list.push_back(loops[0]);
      loop_list.push_back(loops[1]);
    }
    ASSERT_EQ(loop_list.size(), loop_count);
    for (int i = 0; i < loop_count; i++) {
      if (test_cfg & (1 << i)) {
        loop_list[i]->set_parallel();
      }
    }
    loop_nest.prepareForCodegen();
    Stmt* stmt = loop_nest.root_stmt();
    LLVMCodeGen cg(stmt, {t4});

    PaddedBuffer<float> t4_v(M, N, "t4_v");
    std::vector<void*> args({t4_v.data()});
    int value = cg.value<int>(args);
    ASSERT_EQ(value, 0);
    PaddedBuffer<float> t4_ref(M, N, "t4_ref");
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        t4_ref(m, n) = (m + 1) * (n + 2) + m + n;
      }
    }
    ExpectAllNear(t4_v, t4_ref, 1e-5);
    int count = counter.elapsed_value();
    if (test_cfg == 0) {
      ASSERT_EQ(count, 0);
    } else {
      ASSERT_GT(count, 0);
    }
  }
}

TEST(LLVM, VectorizedGEMM) {
  KernelScope ks;

  int M = 32;
  int N = 32;
  int K = 48;

  Placeholder AP(BufHandle("A", {M, K}, kFloat));
  Placeholder BP(BufHandle("B", {K, N}, kFloat));
  Tensor* CT = Reduce(
      "gemm",
      {{M, "M"}, {N, "N"}},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {{K, "K"}});
  LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* m = loops[0];
    For* mo;
    For* mi;
    loop.splitWithMask(m, 16, &mo, &mi);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* n = loops[2];
    For* no;
    For* ni;
    loop.splitWithMask(n, 16, &no, &ni);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* mi = loops[1];
    For* no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* ni = loops[3];
    For* k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    For* mi = loops[2];
    For* k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto loops = NodeFinder<For>::find(loop.root_stmt());
    loop.vectorize(loops[3]);
    loop.vectorize(loops.back());
  }

  loop.prepareForCodegen();

  Stmt* s = loop.root_stmt();
  s = IRSimplifier::simplify(s);
  LLVMCodeGen cg(s, {AP, BP, CT});

  PaddedBuffer<float> a_v(M, K, "a_v");
  PaddedBuffer<float> b_v(K, N, "b_v");
  PaddedBuffer<float> c_v(M, N, "c_v");
  PaddedBuffer<float> c_ref(M, N, "c_ref");

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      c_ref(m, n) = 0.f;
      for (int k = 0; k < K; k++) {
        c_ref(m, n) += a_v(m, k) * b_v(k, n);
      }
    }
  }

  cg.call({a_v, b_v, c_v});

  ExpectAllNear(c_v, c_ref, 1e-5);
}

} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM

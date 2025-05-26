#ifdef TORCH_ENABLE_LLVM
#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>
#include <torch/csrc/jit/testing/file_check.h>

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
  auto a = IntImm::make(2);
  auto b = Cast::make(kFloat, a);
  LLVMExprEval cg(b, {});
  ASSERT_EQ(cg.value<float>(), 2.0);
}

TEST(LLVM, FloatToIntCastTest) {
  auto a = FloatImm::make(2.0);
  auto b = Cast::make(kInt, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int>(), 2);
}

TEST(LLVM, IntToLongCastTest) {
  auto a = IntImm::make(12345);
  auto b = Cast::make(kLong, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int64_t>(), 12345);
}

TEST(LLVM, ByteToCharCastTest) {
  auto a = ByteImm::make(250);
  auto b = Cast::make(kChar, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int8_t>(), (int8_t)250);
}

TEST(LLVM, HalfToLongCastTest) {
  auto a = HalfImm::make(2.0);
  auto b = Cast::make(kLong, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int64_t>(), 2);
}

TEST(LLVM, ByteToDoubleCastTest) {
  auto a = ByteImm::make(2);
  auto b = Cast::make(kDouble, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<double>(), 2);
}

TEST(LLVM, FloatToByteCastTest) {
  auto a = FloatImm::make(254.0);
  auto b = Cast::make(kByte, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<uint8_t>(), 254);
}

TEST(LLVM, FloatToCharCastTest) {
  auto a = FloatImm::make(-2.0);
  auto b = Cast::make(kChar, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<int8_t>(), -2);
}

TEST(LLVM, ByteToFloatCastTest) {
  auto a = ByteImm::make(254);
  auto b = Cast::make(kFloat, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<float>(), 254.0);
}

TEST(LLVM, CharToFloatCastTest) {
  auto a = CharImm::make(-2);
  auto b = Cast::make(kFloat, a);
  LLVMExprEval cg(b);
  ASSERT_EQ(cg.value<float>(), -2.0);
}

TEST(LLVM, BitCast) {
  /* constexpr int16_t ref16 = 1337; */
  constexpr int32_t ref32 = 1337;
  constexpr int64_t ref64 = 1337;
  constexpr float reff32 = 1337.0f;
  constexpr double reff64 = 1337.0f;

  // this is broken
  /*{
    at::Half k_;
    at::Half* k = &k_;
    *reinterpret_cast<int16_t*>(k) = ref16;
    auto a = HalfImm::make(k);
    auto b = BitCast::make(kShort, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<int16_t>(), ref16);
  }*/

  {
    float k = raw_bitcast<float>(ref32);
    auto a = FloatImm::make(k);
    auto b = BitCast::make(kInt, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<int32_t>(), ref32);
  }

  {
    double k = raw_bitcast<double>(ref64);
    auto a = DoubleImm::make(k);
    auto b = BitCast::make(kLong, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<int64_t>(), ref64);
  }

  {
    int64_t k = raw_bitcast<int64_t>(reff64);
    auto a = LongImm::make(k);
    auto b = BitCast::make(kDouble, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<double>(), reff64);
  }

  {
    int32_t k = raw_bitcast<int32_t>(reff32);
    auto a = IntImm::make(k);
    auto b = BitCast::make(kFloat, a);
    LLVMExprEval cg(b);
    ASSERT_EQ(cg.value<float>(), reff32);
  }
}

TEST(LLVM, fastLogFloat) {
  const int kTotalSize = 128 * 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, fast_log(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  LLVMCodeGen ir_eval(stmt, {a_buf, b_buf});
  ir_eval.call({a_v, b_v});

  for (const auto i : c10::irange(kTotalSize)) {
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
  BufHandle a("A", {1}, kFloat);
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
  BufHandle a("A", {1}, kFloat);
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
  BufHandle a("A", {1}, kDouble);
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
  BufHandle a("A", {32}, kFloat);
  std::vector<int32_t> v(5);
  std::vector<void*> args({v.data()});
  auto rv = IntImm::make(0);
  LLVMExprEval cg(rv, {a});
  ASSERT_EQ(cg.value<int>(args), 0);
}

TEST(LLVM, BlockTest) {
  BufHandle a("A", {32}, kInt);
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
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
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
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  BufHandle c("C", {1}, kInt);
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
  BufHandle x("X", {1}, kInt);
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
  BufHandle x("X", {1}, kInt);
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
  BufHandle x("X", {1}, kInt);
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
  constexpr int M = 3;
  constexpr int N = 64;
  BufHandle a("a", {M, N}, kFloat);
  BufHandle b("b", {M, N}, kFloat);
  BufHandle c("c", {M, N}, kFloat);
  VarHandle m("m", kInt);
  VarHandle n("n", kInt);
  StmtPtr s = For::make(
      m,
      0,
      M,
      Store::make(
          c,
          {Ramp::make(m * 64, 1, 64)},
          Load::make({kFloat, 64}, a, {Ramp::make(m * 64, 1, 64)}) *
              Load::make({kFloat, 64}, b, {Ramp::make(m * 64, 1, 64)})));
  LLVMCodeGen cg(s, {a, b, c});
}

TEST(LLVM, VecLoadStoreTest) {
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  std::vector<int32_t> a_buffer = {1, 1, 1, 1};
  std::vector<int32_t> b_buffer = {2, 2, 2, 2};

  auto store = b.store({Ramp::make(0, 1, 4)}, a.load({Ramp::make(0, 1, 4)}));
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

#define FLOAT_INTRINSICS_TEST(Name, Lanes)                                   \
  TEST(LLVM, VecFloat_##Name##Lane##Lanes##Test) {                           \
    BufHandle a("A", {1}, kFloat);                                           \
    BufHandle b("B", {1}, kFloat);                                           \
    float val = 0.5f;                                                        \
    std::vector<float> a_buffer(Lanes, val);                                 \
    std::vector<float> b_buffer(Lanes, val);                                 \
    auto store = b.store(                                                    \
        {Ramp::make(0, 1, Lanes)}, Name(a.load({Ramp::make(0, 1, Lanes)}))); \
    LLVMCodeGen cg(store, {a, b});                                           \
    std::vector<void*> args({a_buffer.data(), b_buffer.data()});             \
    ASSERT_EQ(cg.value<int>(args), 0);                                       \
    for (const auto i : c10::irange(Lanes)) {                                \
      ASSERT_FLOAT_EQ(a_buffer[i], val);                                     \
    }                                                                        \
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

#define DOUBLE_INTRINSICS_TEST(Name, Lanes)                                  \
  TEST(LLVM, VecDouble_##Name##Lane##Lanes##Test) {                          \
    BufHandle a("A", {1}, kDouble);                                          \
    BufHandle b("B", {1}, kDouble);                                          \
    float val = 0.5f;                                                        \
    std::vector<double> a_buffer(Lanes, val);                                \
    std::vector<double> b_buffer(Lanes, val);                                \
    auto store = b.store(                                                    \
        {Ramp::make(0, 1, Lanes)}, Name(a.load({Ramp::make(0, 1, Lanes)}))); \
    LLVMCodeGen cg(store, {a, b});                                           \
    std::vector<void*> args({a_buffer.data(), b_buffer.data()});             \
    ASSERT_EQ(cg.value<int>(args), 0);                                       \
    for (const auto i : c10::irange(Lanes)) {                                \
      ASSERT_FLOAT_EQ(a_buffer[i], val);                                     \
    }                                                                        \
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
  BufHandle a("A", {1}, kInt);

  Tensor c = Compute("c", {4}, [&](const VarHandle& i) { return a.load(i); });

  BufHandle c_buf(c.buf());
  LoopNest l({c});
  StmtPtr s = l.root_stmt();
  ASSERT_TRUE(LoopNest::vectorize(to<For>(to<Block>(s)->front())));

  ASSERT_TRUE(to<For>(to<Block>(s)->front()) == nullptr);

  LLVMCodeGen cg(s, {a, c_buf});

  std::vector<int> a_vec(4, 21);
  std::vector<int> c_vec(4, 0);
  std::vector<void*> args({a_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 21);
}

TEST(LLVM, VectorizeBitCast) {
  BufHandle a("A", {128}, kInt);

  Tensor c = Compute("c", {128}, [&](const VarHandle& i) {
    return bitcast<float>(a.load(i));
  });

  BufHandle c_buf(c.buf());
  LoopNest l({c});
  StmtPtr s = l.root_stmt();
  ASSERT_TRUE(LoopNest::vectorize(to<For>(to<Block>(s)->front())));
  ASSERT_TRUE(to<For>(to<Block>(s)->front()) == nullptr);

  LLVMCodeGen cg(s, {a, c_buf});

  std::vector<int> a_vec(128);
  std::vector<float> c_vec(128);
  for (const auto i : c10::irange(128)) {
    a_vec[i] = raw_bitcast<int>(1337.f);
  }
  std::vector<void*> args({a_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 1337.f);
}

TEST(LLVM, MemcpyTest) {
  constexpr int N = 32;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
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
  constexpr int N = 32;
  BufHandle b("B", {N}, kInt);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  std::vector<float> a_buffer(N, 10.0f);
  std::vector<float> b_buffer(N, 2.0f);

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N / 4,
      b.store(
          {Ramp::make(i * 4, 1, 4)}, log10(a.load({Ramp::make(i * 4, 1, 4)}))));

  LLVMCodeGen cg(expr, {a, b});

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, 10.0f);
  assertAllEqual(b_buffer, 1.0f);
}

TEST(LLVM, ElemwiseLog1pFloat) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  std::vector<float> a_buffer(N, expf(3.0f) - 1);
  std::vector<float> b_buffer(N, 42.0f);

  VarHandle i("i", kInt);
  auto expr = For::make(
      i,
      0,
      N / 4,
      b.store(
          {Ramp::make(i * 4, 1, 4)}, log1p(a.load({Ramp::make(i * 4, 1, 4)}))));

  LLVMCodeGen cg(expr, {a, b});

  std::vector<void*> args({a_buffer.data(), b_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  assertAllEqual(a_buffer, expf(3.0f) - 1);
  ExpectAllNear(b_buffer, 3.0f, 1e-5f);
}

TEST(LLVM, ElemwiseMaxInt) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kFloat);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
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
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectFloatEQ) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kFloat);
  BufHandle b("B", {N}, kFloat);
  BufHandle c("C", {N}, kInt);
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
  constexpr int N = 1024;
  BufHandle a("A", {N}, kByte);
  BufHandle b("B", {N}, kByte);
  BufHandle c("C", {N}, kInt);
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
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectByteGE) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kByte);
  BufHandle b("B", {N}, kByte);
  BufHandle c("C", {N}, kInt);
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
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectByteLT) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kByte);
  BufHandle b("B", {N}, kByte);
  BufHandle c("C", {N}, kInt);
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
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, CompareSelectByteLE) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kByte);
  BufHandle b("B", {N}, kByte);
  BufHandle c("C", {N}, kInt);
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
  for (const auto i : c10::irange(N)) {
    ASSERT_EQ(c_ref[i], c_buffer[i]);
  }
}

TEST(LLVM, StoreFloat) {
  BufHandle result("result", {1}, kFloat);
  std::vector<float> result_buffer = {0.0f};
  auto expr = result.store({0}, FloatImm::make(3.14f));
  LLVMCodeGen cg(expr, {result});
  std::vector<void*> args({result_buffer.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  ASSERT_EQ(result_buffer[0], 3.14f);
}

TEST(LLVM, SimpleMath01) {
  const int N = 1024;
  Tensor tensor = Compute(
      "f", {N}, [](const VarHandle& i) { return cast<float>(i * i + 1); });
  LoopNest l({tensor});
  StmtPtr stmt = l.root_stmt();
  BufHandle f_buf(tensor.buf());
  LLVMCodeGen cg(stmt, {f_buf});

  PaddedBuffer<float> f_v(N, "f_v");
  std::vector<void*> args({f_v.data()});
  int value = cg.value<int>(args);
  ASSERT_EQ(value, 0);
  PaddedBuffer<float> f_ref(N, "f_ref");
  for (const auto i : c10::irange(N)) {
    f_ref(i) = i * i + 1;
  }
  ExpectAllNear(f_v, f_ref, 1e-5);
}

TEST(LLVM, ComputeMul) {
  const int N = 1024;
  BufHandle a("a", {N}, kFloat);
  BufHandle b("b", {N}, kFloat);
  Tensor c = Compute(
      "c", {N}, [&](const VarHandle& i) { return a.load(i) * b.load(i); });

  BufHandle c_buf(c.buf());
  LoopNest l({c});
  StmtPtr s = l.root_stmt();

  LLVMCodeGen cg(s, {a, b, c_buf});

  std::vector<float> a_vec(N, 21.0f);
  std::vector<float> b_vec(N, 2.0f);
  std::vector<float> c_vec(N, 0.0f);
  std::vector<void*> args({a_vec.data(), b_vec.data(), c_vec.data()});
  ASSERT_EQ(cg.value<int>(args), 0);
  assertAllEqual(c_vec, 42.0f);
}

TEST(LLVM, BroadcastAdd) {
  const int M = 32;
  const int N = 1024;
  BufHandle a("a", {M, N}, kFloat);
  BufHandle b("b", {N}, kFloat);
  Tensor c = Compute("c", {M, N}, [&](const VarHandle& i, const VarHandle& j) {
    return a.load(i, j) + b.load(j);
  });

  BufHandle c_buf(c.buf());
  LoopNest l({c});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();

  LLVMCodeGen cg(s, {a, b, c_buf});

  std::vector<float> av(M * N);
  std::iota(av.begin(), av.end(), 0);
  std::vector<float> bv(N);
  std::iota(bv.begin(), bv.end(), 0);
  std::vector<float> cv(M * N, 0);
  std::vector<void*> args({av.data(), bv.data(), cv.data()});
  ASSERT_EQ(cg.value<int>(args), 0);

  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      ASSERT_EQ(cv[i * N + j], av[i * N + j] + bv[j]);
    }
  }
}

TEST(LLVM, BitwiseOps) {
  auto a = IntImm::make(59);
  auto b = IntImm::make(11);
  auto c = IntImm::make(101);
  auto d = IntImm::make(2);

  ExprHandle f = (((a ^ (b << 1)) & c) >> 2) | d;
  LLVMExprEval cg(f);

  ASSERT_EQ(cg.value<int>(), 11);
}

TEST(LLVM, ArithmeticRightShift) {
  auto a = CharImm::make(-4);
  auto b = CharImm::make(1);
  ExprHandle f = a >> b;
  LLVMExprEval cg(f);
  ASSERT_EQ(cg.value<int8_t>(), -2);
}

TEST(LLVM, LogicalRightShift) {
  auto a = ByteImm::make(0xfc);
  auto b = ByteImm::make(1);
  ExprHandle f = a >> b;
  LLVMExprEval cg(f);
  ASSERT_EQ(cg.value<uint8_t>(), 0x7e);
}

TEST(LLVM, DynamicShapeAdd) {
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);
    BufHandle a("a", {n}, kFloat);
    BufHandle b("b", {n}, kFloat);
    BufHandle c("c", {n}, kFloat);
    VarHandle i("i", kInt);
    StmtPtr s = For::make(i, 0, n, c.store({i}, a.load(i) + b.load(i)));
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
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);
    BufHandle a("a", {n}, kFloat);
    BufHandle b("b", {n}, kFloat);
    BufHandle c("c", {n}, kFloat);
    VarHandle i("i", kInt);
    StmtPtr s = For::make(i, 0, n, c.store({i}, a.load(i) + b.load(i)));
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
  auto testWithSize = [](int32_t size) {
    VarHandle n("n", kInt);
    BufHandle a("a", {n}, kFloat);
    BufHandle b("b", {n}, kFloat);
    Tensor c = Compute(
        "c", {n}, [&](const VarHandle& i) { return a.load(i) + b.load(i); });
    LoopNest l({c});
    StmtPtr s = l.root_stmt();
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
  auto testWithSize = [](int32_t M, int32_t N) {
    VarHandle m("m", kInt);
    VarHandle n("n", kInt);
    BufHandle a("a", {m, n}, kFloat);
    BufHandle b("b", {m, n}, kFloat);
    Tensor c =
        Compute("c", {m, n}, [&](const VarHandle& i, const VarHandle& j) {
          return a.load(i, j) + b.load(i, j);
        });
    LoopNest l({c});
    l.prepareForCodegen();
    StmtPtr s = l.root_stmt();
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
  StmtPtr s = alloc<Block>(std::vector<StmtPtr>({}));

  LLVMCodeGen cg(s, {});
  cg.call({});
  // Just don't crash.
}

TEST(LLVM, EliminatedStmt) {
  BufHandle a("a", {1}, kFloat);

  Tensor c = Compute("c", {0}, [&](const VarHandle& m) { return m; });

  LoopNest l({c});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();
  s = IRSimplifier::simplify(s);
  LLVMCodeGen cg(s, {a, c});
  std::vector<float> aData(1, 1.0f);
  std::vector<float> cData(0, 0.0f);
  cg.call({aData, cData});
}

TEST(LLVM, SimpleReduction) {
  int M = 128;
  int N = 64;

  BufHandle a("a", {1, M, N}, kFloat);

  Tensor b = Reduce("sum", {1}, Sum(), a, {M, N});
  LoopNest loop({b});

  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  LLVMCodeGen cg(s, {a, b});

  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  cg.call({a_v, b_v});

  ExpectAllNear(b_v, b_ref, 1e-5);
}

TEST(LLVM, RFactorReduction) {
  int M = 128;
  int N = 64;

  BufHandle a("a", {1, M, N}, kFloat);

  Tensor b = Reduce("sum", {1}, Sum(), a, {M, N});
  LoopNest loop({b});

  std::vector<ForPtr> loops = loop.getLoopStmtsFor(b);
  ForPtr loop_m = loops.at(1);
  ForPtr loop_n = loops.at(2);
  loop.reorderAxis(loop_m, loop_n);

  loops = loop.getLoopStmtsFor(b);
  loop_m = loops.at(2);
  loop_n = loops.at(1);
  auto b_body = loop.getAllWritesToBuf(b.buf())[1];
  ASSERT_TRUE(loop.rfactor(b_body, loop_n));

  loop.prepareForCodegen();
  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);

  LLVMCodeGen cg(s, {a, b});

  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  cg.call({a_v, b_v});

  ExpectAllNear(b_v, b_ref, 1e-5);
}

TEST(LLVM, RFactorVectorizedReduction) {
  int M = 128;
  int N = 64;

  BufHandle a("a", {1, M, N}, kFloat);

  Tensor b = Reduce("sum", {1}, Sum(), a, {M, N});
  LoopNest loopnest({b});
  std::vector<ForPtr> loops = loopnest.getLoopStmtsFor(b);
  // Reorder n and m loops
  loopnest.reorderAxis(loops.at(1), loops.at(2));
  auto b_body = loopnest.getAllWritesToBuf(b.buf()).at(1);
  auto all_loops = loopnest.getAllLoopNestsWritingToBuf(b.buf());
  ASSERT_TRUE(all_loops.size() == 2 && all_loops[1].size() == 3);
  ASSERT_TRUE(loopnest.rfactor(b_body, all_loops[1][1]));
  auto distributed_loops = loopnest.distributeLoop(all_loops[1][1]);

  // Vectorize initializer of rfac_buf
  ASSERT_TRUE(LoopNest::vectorize(distributed_loops[0]));
  // Vectorize producer of rfac_buf
  ASSERT_TRUE(LoopNest::vectorize(distributed_loops[1]));
  loopnest.simplify();

  loopnest.prepareForCodegen();

  StmtPtr s = IRSimplifier::simplify(loopnest.root_stmt());
  LLVMCodeGen cg(s, {a, b});

  PaddedBuffer<float> a_v(1, M, N, "a_v");
  PaddedBuffer<float> b_v(1, "b_v");
  PaddedBuffer<float> b_ref(1, "b_ref");

  b_ref(0) = 0;
  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N)) {
      int v = i + j;
      a_v(0, i, j) = v;
      b_ref(0) += v;
    }
  }

  cg.call({a_v, b_v});

  ExpectAllNear(b_v, b_ref, 1e-5);
}

template <bool outer, bool inner>
static void testSimpleParallel() {
  // Compute a simple operation, and try all loop-axis combination to be
  // parallel or sequential.
  const int M = 4;
  const int N = 6;
  Tensor f = Compute("f", {M, N}, [](const VarHandle& m, const VarHandle& n) {
    return cast<float>(m + n);
  });
  LoopNest loop_nest({f});
  auto const& loops = loop_nest.getLoopStmtsFor(f);
  ForPtr m = loops[0];
  ForPtr n = loops[1];
  if (outer) {
    m->set_parallel();
  }
  if (inner) {
    n->set_parallel();
  }
  loop_nest.prepareForCodegen();
  StmtPtr stmt = loop_nest.root_stmt();
  LLVMCodeGen cg(stmt, {f});

  PaddedBuffer<float> f_v(M, N, "f_v");
  std::vector<void*> args({f_v.data()});
  int value = cg.value<int>(args);
  ASSERT_EQ(value, 0);
  PaddedBuffer<float> f_ref(M, N, "f_ref");
  for (const auto m : c10::irange(M)) {
    for (const auto n : c10::irange(N)) {
      f_ref(m, n) = m + n;
    }
  }
  ExpectAllNear(f_v, f_ref, 1e-5);
}

TEST(LLVM, SimpleParallelSS) {
  testSimpleParallel<false, false>();
}
TEST(LLVM, SimpleParallelSP) {
  testSimpleParallel<false, true>();
}
TEST(LLVM, SimpleParallelPS) {
  testSimpleParallel<true, false>();
}
TEST(LLVM, SimpleParallelPP) {
  testSimpleParallel<true, true>();
}

TEST(LLVM, CompositeParallel) {
  int loop_count = 6;
  int test_count = 1 << loop_count;
  // Compute a composite operation, and try all loop-axis combination to be
  // parallel or sequential.
  for (const auto test_cfg : c10::irange(test_count)) {
    int M = 5;
    int N = 7;
    Tensor t1 = Compute("t1", {M}, [](const VarHandle& m) { return m + 1.f; });
    Tensor t2 = Compute("t2", {N}, [](const VarHandle& n) { return n + 2.f; });
    Tensor t3 =
        Compute("t3", {M, N}, [=](const VarHandle& m, const VarHandle& n) {
          return t1.load(m) * t2.load(n);
        });
    Tensor t4 =
        Compute("t4", {M, N}, [=](const VarHandle& m, const VarHandle& n) {
          return t3.load(m, n) + m + n;
        });
    LoopNest loop_nest({t4}, {t1, t2, t3, t4});
    std::vector<ForPtr> loop_list;
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
    for (const auto i : c10::irange(loop_count)) {
      if (test_cfg & (1 << i)) {
        loop_list[i]->set_parallel();
      }
    }
    loop_nest.prepareForCodegen();
    StmtPtr stmt = loop_nest.root_stmt();
    LLVMCodeGen cg(stmt, {t4});

    PaddedBuffer<float> t4_v(M, N, "t4_v");
    std::vector<void*> args({t4_v.data()});
    int value = cg.value<int>(args);
    ASSERT_EQ(value, 0);
    PaddedBuffer<float> t4_ref(M, N, "t4_ref");
    for (const auto m : c10::irange(M)) {
      for (const auto n : c10::irange(N)) {
        t4_ref(m, n) = (m + 1) * (n + 2) + m + n;
      }
    }
    ExpectAllNear(t4_v, t4_ref, 1e-5);
  }
}

TEST(LLVM, VectorizedGEMM) {
  int M = 32;
  int N = 32;
  int K = 48;

  BufHandle AP("A", {M, K}, kFloat);
  BufHandle BP("B", {K, N}, kFloat);
  Tensor CT = Reduce(
      "gemm",
      {M, N},
      Sum(),
      [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
        return AP.load(m, k) * BP.load(k, n);
      },
      {K});
  LoopNest loop({CT});

  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr m = loops[0];
    loop.splitWithMask(m, 16);
  }
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr n = loops[2];
    loop.splitWithMask(n, 16);
  }
  // mo, mi, no, ni, k ->
  // mo, no, mi, ni, k
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr mi = loops[1];
    ForPtr no = loops[2];
    loop.reorderAxis(mi, no);
  }
  // mo, no, mi, ni, k ->
  // mo, no, mi, k, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr ni = loops[3];
    ForPtr k = loops[4];
    loop.reorderAxis(ni, k);
  }
  // mo, no, mi, k, ni ->
  // mo, no, k, mi, ni
  {
    auto const& loops = loop.getLoopStmtsFor(CT);
    ForPtr mi = loops[2];
    ForPtr k = loops[3];
    loop.reorderAxis(mi, k);
  }
  {
    auto loops = NodeFinder<For>::find(loop.root_stmt());
    ASSERT_TRUE(LoopNest::vectorize(loops[3]));
    ASSERT_TRUE(LoopNest::vectorize(loops.back()));
  }

  loop.prepareForCodegen();

  StmtPtr s = loop.root_stmt();
  s = IRSimplifier::simplify(s);
  LLVMCodeGen cg(s, {AP, BP, CT});

  PaddedBuffer<float> a_v(M, K, "a_v");
  PaddedBuffer<float> b_v(K, N, "b_v");
  PaddedBuffer<float> c_v(M, N, "c_v");
  PaddedBuffer<float> c_ref(M, N, "c_ref");

  for (const auto m : c10::irange(M)) {
    for (const auto n : c10::irange(N)) {
      c_ref(m, n) = 0.f;
      for (const auto k : c10::irange(K)) {
        c_ref(m, n) += a_v(m, k) * b_v(k, n);
      }
    }
  }

  cg.call({a_v, b_v, c_v});

  ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(LLVM, CallRaw) {
  const int M = 32;
  VarHandle N("N", kInt);
  BufHandle a("a", {M, N}, kFloat);
  BufHandle b("b", {N}, kFloat);
  Tensor c = Compute("c", {M, N}, [&](const VarHandle& i, const VarHandle& j) {
    return a.load(i, j) + b.load(j);
  });

  LoopNest l({c});
  l.prepareForCodegen();
  StmtPtr s = l.root_stmt();

  int32_t N_value = 1024;
  std::vector<float> av(M * N_value);
  std::iota(av.begin(), av.end(), 0);
  std::vector<float> bv(N_value);
  std::iota(bv.begin(), bv.end(), 0);
  std::vector<float> cv(M * N_value, 0);
  std::vector<void*> args({av.data(), bv.data(), cv.data(), &N_value});

  LLVMCodeGen cg(s, {a, b, BufHandle(c.buf()), N});
  cg.call_raw(args);

  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N_value)) {
      ASSERT_EQ(cv[i * N_value + j], av[i * N_value + j] + bv[j]);
    }
  }

  SimpleIREvaluator eval(s, {a, b, BufHandle(c.buf()), N});
  eval.call_raw(args);

  for (const auto i : c10::irange(M)) {
    for (const auto j : c10::irange(N_value)) {
      ASSERT_EQ(cv[i * N_value + j], av[i * N_value + j] + bv[j]);
    }
  }
}

TEST(LLVM, CustomTarget) {
  constexpr int M = 16;
  BufHandle a("a", {M}, kFloat);
  BufHandle b("b", {M}, kFloat);
  BufHandle c("c", {M}, kFloat);
  Tensor d = Compute("d", {M}, [&](const VarHandle& m) {
    return a.load(m) * b.load(m) + c.load(m);
  });
  LoopNest nest({d});
  nest.prepareForCodegen();
  auto cg = LLVMCodeGenBuilder(nest.root_stmt(), {a, b, c, d})
                .triple("i686-elf")
                .cpu("i386")
                .build();
  std::ostringstream ss;
  ss << cg->getCodeText("asm");
  torch::jit::testing::FileCheck()
      .check("fadds")
      ->check("fmuls")
      ->check_not("vfmadd")
      ->run(ss.str());
}

TEST(LLVM, CodeGenKernelFuncName) {
  BufHandle a("A", {1}, kInt);
  BufHandle b("B", {1}, kInt);
  std::vector<int32_t> a_buffer = {42};
  std::vector<int32_t> b_buffer = {-11};
  auto store = b.store({0}, a.load(0));

  {
    LLVMCodeGen cg(store, {a, b});
    // Check that the kernel function name used by LLVMCodeGen
    // is not empty.
    ASSERT_NE(cg.kernel_func_name(), "");
  }

  {
    LLVMCodeGen cg(store, {a, b}, at::kCPU, "new_func");
    // Check that the kernel function name used by LLVMCodeGen
    // is the one that was given above.
    ASSERT_EQ(cg.kernel_func_name(), "new_func");
  }
}

} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM

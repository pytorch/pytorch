#include <gtest/gtest.h>

#include <test/cpp/tensorexpr/test_base.h>

#include <c10/util/irange.h>
#include <test/cpp/tensorexpr/padded_buffer.h>
#include <test/cpp/tensorexpr/test_utils.h>
#include <torch/csrc/jit/tensorexpr/eval.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_verifier.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace jit {
using namespace torch::jit::tensorexpr;

using SimpleIRExprEval = ExprEval<SimpleIREvaluator>;

TEST(Expr, BasicValueTest) {
  ExprHandle a = IntImm::make(2), b = IntImm::make(3);
  ExprHandle c = Add::make(a, b);
  SimpleIRExprEval eval(c);
  ASSERT_EQ(eval.value<int>(), 5);
}

TEST(Expr, BasicValueTest02) {
  ExprHandle a(2.0f);
  ExprHandle b(3.0f);
  ExprHandle c(4.0f);
  ExprHandle d(5.0f);
  ExprHandle f = (a + b) - (c + d);
  SimpleIRExprEval eval(f);
  ASSERT_EQ(eval.value<float>(), -4.0f);
}

TEST(Expr, IsChannelsLastContiguous) {
  std::vector<VarHandle> vars = {
      VarHandle("var1", kLong),
      VarHandle("var2", kLong),
      VarHandle("var3", kLong),
      VarHandle("var4", kLong),
      VarHandle("var5", kLong)};

  // {
  //   key: ndims,
  //   value: [
  //     ...
  //     [dim_2, dim_1, ..., dim_n]
  //   ]
  // }
  using shapGenInfo = std::unordered_map<int, std::vector<std::vector<int>>>;

  // {
  //   size: [ExprHandle_1, ExprHandle_2, ..., ExprHandle_n],
  //   strides: [
  //     ...
  //     [ExprHandle_x, ExprHandle_y, ..., ExprHandle_z]
  //   ]
  // }
  using shapeInfo =
      std::pair<std::vector<ExprHandle>, std::vector<std::vector<ExprHandle>>>;

  std::vector<int> dims = {3, 4, 5};

  std::unordered_map<int, std::vector<ExprHandle>> dims_expr_vec_conf = {
      {3, std::vector<ExprHandle>(vars.begin(), vars.begin() + 2)},
      {4, std::vector<ExprHandle>(vars.begin(), vars.begin() + 3)},
      {5, std::vector<ExprHandle>(vars.begin(), vars.begin() + 4)},
  };

  shapGenInfo channels_last_cont_shape_conf = {
      {3, {{1, 2, 0}}}, {4, {{1, 3, 2, 0}}}, {5, {{1, 4, 3, 2, 0}}}};
  shapGenInfo channels_last_non_cont_shape_conf = {
      {3, {{2, 1, 0}, {1, 0, 2}}},
      {4, {{3, 1, 2, 0}, {1, 2, 3, 0}, {1, 0, 2, 3}}},
      {5, {{4, 3, 2, 1, 0}, {1, 3, 2, 4, 0}, {1, 4, 3, 2, 0}}}};

  shapGenInfo cont_shape_conf = {
      {3, {{0, 1, 2}}}, {4, {{0, 1, 2, 3}}}, {5, {{0, 1, 2, 3, 4}}}};

  auto shape_gen_fn = [dims_expr_vec_conf](
                          int ndims, shapGenInfo shape_gen_info) -> shapeInfo {
    auto dims_expr_vec = dims_expr_vec_conf.at(ndims);
    std::vector<std::vector<ExprHandle>> strides_expr_vec;
    for (size_t i = 0; i < strides_expr_vec.size(); i++) {
      strides_expr_vec[i].resize(ndims);
    }

    auto stride_gen_fn = [](int indicator, ExprHandle a, ExprHandle b) {
      if (indicator % 2 == 0) {
        return a * b;
      } else {
        return b * a;
      }
    };

    auto stride_order_vec = shape_gen_info.at(ndims);
    for (size_t i = 0; i < strides_expr_vec.size(); i++) {
      auto stride_order = stride_order_vec[i];

      strides_expr_vec[i][stride_order[0]] = 1;
      for (size_t j = 1; j < stride_order.size(); j++) {
        auto cur_dim_idx = stride_order[j];
        auto adjacent_dim_idx = stride_order[j - 1];

        strides_expr_vec[i][cur_dim_idx] = stride_gen_fn(
            i,
            dims_expr_vec[adjacent_dim_idx],
            strides_expr_vec[i][adjacent_dim_idx]);
      }
    }

    return {dims_expr_vec, strides_expr_vec};
  };

  auto check_channels_last_fn = [](int ndims, BufHandle buf_handle) -> bool {
    if (ndims == 3) {
      return buf_handle.is_channels_last_1d_contiguous();
    } else if (ndims == 4) {
      return buf_handle.is_contiguous(at::MemoryFormat::ChannelsLast);
    } else {
      return buf_handle.is_contiguous(at::MemoryFormat::ChannelsLast3d);
    }
  };

  // channels-last contiguous
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], channels_last_cont_shape_conf);
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      ASSERT_EQ(check_channels_last_fn(dims[i], buf_handle), true);
    }
  }

  // channels-last non-contiguous
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], channels_last_non_cont_shape_conf);
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      ASSERT_EQ(check_channels_last_fn(dims[i], buf_handle), false);
    }
  }

  // contiguous
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], cont_shape_conf);
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      ASSERT_EQ(buf_handle.is_contiguous(), true);
    }
  }

  // non-contiguous
  for (size_t i = 0; i < dims.size(); i++) {
    auto shape_info = shape_gen_fn(dims[i], channels_last_cont_shape_conf);
    for (size_t j = 0; j < shape_info.second.size(); j++) {
      BufHandle buf_handle("a", shape_info.first, shape_info.second[j], kFloat);
      ASSERT_EQ(buf_handle.is_contiguous(), false);
    }
  }
}

TEST(Expr, LetTest01) {
  VarHandle x("x", kFloat);
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle(3.f));
  ASSERT_EQ(eval.value<float>(), 2 + (3 * 3 + 4));
}

TEST(Expr, LetTest02) {
  VarHandle x("x", kFloat);
  VarHandle y("y", kFloat);
  ExprHandle body =
      ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f) * y);
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle(3.f));
  eval.bindVar(y, ExprHandle(6.f));
  ASSERT_EQ(eval.value<float>(), 2 + (3 * 3 + 4 * 6));
}

TEST(Expr, LetStmtTest01) {
  BufHandle a_buf("a", {1}, kFloat);
  BufHandle b_buf("b", {1}, kFloat);

  ExprHandle load_a = a_buf.load(0);
  VarHandle var = VarHandle("v", kFloat);
  StmtPtr let_store = Let::make(var, load_a);
  StmtPtr store_b = b_buf.store({0}, var);
  BlockPtr block = Block::make({let_store, store_b});

  SimpleIREvaluator eval(block, {a_buf, b_buf});

  PaddedBuffer<float> a_v(1);
  PaddedBuffer<float> b_v(1);
  PaddedBuffer<float> b_ref(1);

  a_v(0) = 23;
  b_ref(0) = a_v(0);
  eval(a_v, b_v);

  ExpectAllNear(b_v, b_ref, 1e-5);
}

TEST(Expr, IntTest) {
  VarHandle x("x", kInt);
  ExprHandle body = ExprHandle(2) + (x * ExprHandle(3) + ExprHandle(4));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle(3));
  ASSERT_EQ(eval.value<int>(), 2 + (3 * 3 + 4));
}

TEST(Expr, FloatTest) {
  VarHandle x("x", kFloat);
  ExprHandle body = ExprHandle(2.f) + (x * ExprHandle(3.f) + ExprHandle(4.f));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle(3.f));
  ASSERT_EQ(eval.value<float>(), 2 + (3 * 3 + 4));
}

TEST(Expr, ByteTest) {
  VarHandle x("x", kByte);
  ExprHandle body = ExprHandle((uint8_t)2) +
      (x * ExprHandle((uint8_t)3) + ExprHandle((uint8_t)4));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle((uint8_t)3));
  ASSERT_EQ(eval.value<uint8_t>(), 2 + (3 * 3 + 4));
}

TEST(Expr, CharTest) {
  VarHandle x("x", kChar);
  ExprHandle body = ExprHandle((int8_t)2) +
      (x * ExprHandle((int8_t)3) + ExprHandle((int8_t)4));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle((int8_t)3));
  ASSERT_EQ(eval.value<int8_t>(), 2 + (3 * 3 + 4));
}

TEST(Expr, ShortTest) {
  VarHandle x("x", kShort);
  ExprHandle body = ExprHandle((int16_t)2) +
      (x * ExprHandle((int16_t)3) + ExprHandle((int16_t)4));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle((int16_t)3));
  ASSERT_EQ(eval.value<int16_t>(), 2 + (3 * 3 + 4));
}

TEST(Expr, LongTest) {
  VarHandle x("x", kLong);
  ExprHandle body = ExprHandle((int64_t)2) +
      (x * ExprHandle((int64_t)3) + ExprHandle((int64_t)4));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle((int64_t)3));
  ASSERT_EQ(eval.value<int64_t>(), 2 + (3 * 3 + 4));
}

TEST(Expr, HalfTest) {
  VarHandle x("x", kHalf);
  ExprHandle body = ExprHandle((at::Half)2) +
      (x * ExprHandle((at::Half)3) + ExprHandle((at::Half)4));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle((at::Half)3));
  ASSERT_EQ(eval.value<at::Half>(), 2 + (3 * 3 + 4));
}

TEST(Expr, DoubleTest) {
  VarHandle x("x", kDouble);
  ExprHandle body = ExprHandle((double)2) +
      (x * ExprHandle((double)3) + ExprHandle((double)4));
  SimpleIRExprEval eval(body);
  eval.bindVar(x, ExprHandle((double)3));
  ASSERT_EQ(eval.value<double>(), 2 + (3 * 3 + 4));
}

TEST(Expr, VectorAdd01) {
  const int kVectorSize = 8;
  const int kVectorCount = 128;
  const int kTotalSize = kVectorSize * kVectorCount;

  BufHandle a_buf("A", {kTotalSize}, kFloat);
  BufHandle b_buf("B", {kTotalSize}, kFloat);
  BufHandle c_buf("C", {kTotalSize}, kFloat);

  /*
  Build the following:
    for (const auto index : c10::irange(kVectorCount)) {
      store(c_buf, ramp(index * 8, 1, 8),
            load(a_buf, ramp(index * 8, 1, 8) +
            load(b_buf, ramp(index * 8, 1, 8))))
    }
  */
  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a =
      a_buf.load({Ramp::make(index * kVectorSize, 1, kVectorSize)});
  ExprHandle load_b =
      b_buf.load({Ramp::make(index * kVectorSize, 1, kVectorSize)});
  ExprHandle value = load_a + load_b;
  StmtPtr store_c =
      c_buf.store({Ramp::make(index * kVectorSize, 1, kVectorSize)}, value);
  StmtPtr stmt = For::make(index, 0, kVectorCount, store_c);

  ASSERT_EQ(load_a.dtype(), Dtype(kFloat, kVectorSize));
  ASSERT_EQ(load_b.dtype(), Dtype(kFloat, kVectorSize));
  ASSERT_EQ(value.dtype(), Dtype(kFloat, kVectorSize));

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> c_ref(kTotalSize);
  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i * i;
    b_v(i) = i * i * 4;
    c_ref(i) = a_v(i) + b_v(i);
  }
  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);
  ExpectAllNear(c_v, c_ref, 1e-5);
}

TEST(Expr, CompareSelectEQ) {
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);
  std::vector<int> c_ref(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));

  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  assertAllEqual(b_buffer, 1);
  assertAllEqual(c_buffer, 1);
}

TEST(Expr, CompareSelectDtypes) {
  // LHS and RHS expressions should have the same dtype, but this dtype could
  // differ from the dtype of the return values (but dtypes of true and false
  // return values should be the same).
  // This test constructs a CompareSelect expression where the input dtype is
  // different from the output dtype and verifies that it works correctly:
  //   result = ((int)lhs == (int)rhs) ? (float)retval1 : (float)retval2
  constexpr int N = 1024;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kFloat);
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<float> c_buffer(N, 0.0f);
  std::vector<float> c_ref(N, 3.14f);

  VarHandle i("i", kInt);
  // C[i] = (A[i] == B[i]) ? 3.14f : 2.78f
  // A and B are int, C is float.
  auto select_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i),
              b.load(i),
              FloatImm::make(3.14f),
              FloatImm::make(2.78f),
              CompareSelectOperation::kEQ)));

  SimpleIREvaluator ir_eval(select_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);
  ASSERT_EQ(c_buffer.size(), N);

  assertAllEqual(a_buffer, 1);
  assertAllEqual(b_buffer, 1);
  ExpectAllNear(c_buffer, c_ref, 1e-7);
}

TEST(Expr, IntrinsicsDtypes) {
  constexpr int N = 256;
  BufHandle a("A", {N}, kDouble);
  BufHandle b("B", {N}, kDouble);
  std::vector<double> a_buffer(N, -10.0);
  std::vector<double> b_buffer(N, 0.0);
  std::vector<double> b_ref(N, 10.0);

  VarHandle i("i", kInt);
  auto abs_expr = For::make(i, 0, N, b.store({i}, tensorexpr::abs(a.load(i))));

  SimpleIREvaluator ir_eval(abs_expr, {a, b});
  ir_eval(a_buffer, b_buffer);

  ASSERT_EQ(a_buffer.size(), N);
  ASSERT_EQ(b_buffer.size(), N);

  assertAllEqual(a_buffer, -10.0);
  ExpectAllNear(b_buffer, b_ref, 1e-7);
}

TEST(Expr, Substitute01) {
  VarPtr x = alloc<Var>("x", kFloat);
  VarPtr y = alloc<Var>("y", kFloat);
  ExprPtr e =
      alloc<Mul>(alloc<Sub>(x, alloc<FloatImm>(1.0f)), alloc<Add>(x, y));

  VarPtr z = alloc<Var>("z", kFloat);
  ExprPtr e2 = Substitute(e, {{x, alloc<Add>(z, alloc<FloatImm>(5.0f))}});
  ExprPtr e2_ref = alloc<Mul>(
      alloc<Sub>(alloc<Add>(z, alloc<FloatImm>(5.0f)), alloc<FloatImm>(1.0f)),
      alloc<Add>(alloc<Add>(z, alloc<FloatImm>(5.0f)), y));
  std::ostringstream oss;
  oss << *e2;
  std::string e2_str = oss.str();

  oss.str("");
  oss << *e2_ref;
  std::string e2_ref_str = oss.str();
  ASSERT_EQ(e2_str, e2_ref_str);
}

TEST(Expr, Math01) {
  ExprHandle v = sin(ExprHandle(1.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "sin(1.f)");

  SimpleIRExprEval eval(v);
  float v_ref = std::sin(1.0f);
  float res = eval.value<float>();
  ASSERT_NEAR(res, v_ref, 1e-6);
}

TEST(Expr, UnaryMath01) {
  struct TestConfig {
    std::function<ExprHandle(const ExprHandle&)> func;
    std::function<float(float)> ref_func;
  };

  std::vector<TestConfig> test_configs = {
      {[](const ExprHandle& v) { return sin(v); },
       [](float v) { return std::sin(v); }},
      {[](const ExprHandle& v) { return sin(v); },
       [](float v) { return std::sin(v); }},
      {[](const ExprHandle& v) { return tan(v); },
       [](float v) { return std::tan(v); }},
      {[](const ExprHandle& v) { return asin(v); },
       [](float v) { return std::asin(v); }},
      {[](const ExprHandle& v) { return acos(v); },
       [](float v) { return std::acos(v); }},
      {[](const ExprHandle& v) { return atan(v); },
       [](float v) { return std::atan(v); }},
      {[](const ExprHandle& v) { return sinh(v); },
       [](float v) { return std::sinh(v); }},
      {[](const ExprHandle& v) { return cosh(v); },
       [](float v) { return std::cosh(v); }},
      {[](const ExprHandle& v) { return tanh(v); },
       [](float v) { return std::tanh(v); }},
      {[](const ExprHandle& v) { return exp(v); },
       [](float v) { return std::exp(v); }},
      {[](const ExprHandle& v) { return tensorexpr::abs(v); },
       [](float v) { return std::fabs(v); }},
      {[](const ExprHandle& v) { return log(v); },
       [](float v) { return std::log(v); }},
      {[](const ExprHandle& v) { return log2(v); },
       [](float v) { return std::log2(v); }},
      {[](const ExprHandle& v) { return log10(v); },
       [](float v) { return std::log10(v); }},
      {[](const ExprHandle& v) { return erf(v); },
       [](float v) { return std::erf(v); }},
      {[](const ExprHandle& v) { return sqrt(v); },
       [](float v) { return std::sqrt(v); }},
      {[](const ExprHandle& v) { return rsqrt(v); },
       [](float v) { return 1.0f / std::sqrt(v); }},
      {[](const ExprHandle& v) { return ceil(v); },
       [](float v) { return std::ceil(v); }},
      {[](const ExprHandle& v) { return floor(v); },
       [](float v) { return std::floor(v); }},
      {[](const ExprHandle& v) { return round(v); },
       [](float v) { return std::round(v); }},
      {[](const ExprHandle& v) { return trunc(v); },
       [](float v) { return std::trunc(v); }},
  };

  for (const TestConfig& test_config : test_configs) {
    const float input_v = 0.8765f;
    ExprHandle v = test_config.func(ExprHandle(input_v));
    float v_ref = test_config.ref_func(input_v);
    SimpleIRExprEval eval(v);
    ASSERT_NEAR(eval.value<float>(), v_ref, 1e-6);
  }

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  for (float input_v : {std::nan("1"), 0., .5}) {
    ExprHandle v = FloatImm::make(input_v);
    SimpleIRExprEval eval(Intrinsics::make(kIsNan, v));
    ASSERT_NEAR(eval.value<int>(), std::isnan(input_v), 0);
  }
}

TEST(Expr, BinaryMath01) {
  struct TestConfig {
    std::function<ExprHandle(const ExprHandle&, const ExprHandle&)> func;
    std::function<float(float, float)> ref_func;
  };

  std::vector<TestConfig> test_configs = {
      {[](const ExprHandle& v1, const ExprHandle& v2) { return pow(v1, v2); },
       [](float v1, float v2) { return std::pow(v1, v2); }},
      {[](const ExprHandle& v1, const ExprHandle& v2) { return fmod(v1, v2); },
       [](float v1, float v2) { return std::fmod(v1, v2); }},
  };

  for (const TestConfig& test_config : test_configs) {
    const float v1 = 0.8765f;
    float v2 = 1.2345f;
    ExprHandle v_expr = test_config.func(ExprHandle(v1), ExprHandle(v2));
    float v_ref = test_config.ref_func(v1, v2);
    SimpleIRExprEval eval(v_expr);
    ASSERT_NEAR(eval.value<float>(), v_ref, 1e-6);
  }
}

TEST(Expr, LogicalOps01) {
  ExprHandle a(23);
  ExprHandle b(11);
  ExprHandle c(0.72f);
  ExprHandle d(0.69f);
  ExprHandle f1 = (a > b) && (c > d);
  ExprHandle f2 = (a > b) && (c < d);
  ExprHandle f3 = (a < b) && (c > d);
  ExprHandle f4 = (a < b) && (c < d);
  ExprHandle f5 = (a < b) || (c > d);
  ExprHandle f6 = (a < b) || (c < d);
  ExprHandle f7 = (a > b) || (c < d);
  ExprHandle f8 = (a > b) || (c > d);

  SimpleIRExprEval eval1(f1);
  SimpleIRExprEval eval2(f2);
  SimpleIRExprEval eval3(f3);
  SimpleIRExprEval eval4(f4);
  SimpleIRExprEval eval5(f5);
  SimpleIRExprEval eval6(f6);
  SimpleIRExprEval eval7(f7);
  SimpleIRExprEval eval8(f8);
  ASSERT_EQ(eval1.value<int>(), 1);
  ASSERT_EQ(eval2.value<int>(), 0);
  ASSERT_EQ(eval3.value<int>(), 0);
  ASSERT_EQ(eval4.value<int>(), 0);
  ASSERT_EQ(eval5.value<int>(), 1);
  ASSERT_EQ(eval6.value<int>(), 0);
  ASSERT_EQ(eval7.value<int>(), 1);
  ASSERT_EQ(eval8.value<int>(), 1);
}

TEST(Expr, LogicalOps02) {
  ExprHandle a(23);
  ExprHandle b(11);
  ExprHandle c(0.72f);
  ExprHandle d(0.72f);

  ExprHandle f1 = (a > b) || (c > d);
  ExprHandle f2 = (a > b) && (c <= d);
  ExprHandle f3 = (a > b) && (c > d);
  ExprHandle ff1 = f1 && f2;
  ExprHandle ff2 = f2 || f3;

  SimpleIRExprEval eval1(ff1);
  SimpleIRExprEval eval2(ff2);
  ASSERT_EQ(eval1.value<int>(), 1);
  ASSERT_EQ(eval2.value<int>(), 1);
}

TEST(Expr, LogicalOps03) {
  ExprHandle a(23);
  ExprHandle b(11);
  ExprHandle c(0.72f);
  ExprHandle d(0.69f);

  // Bool types
  ExprHandle bool_f1 = (a > b) && BoolImm::make(true);
  ExprHandle bool_f2 = (c <= d) || BoolImm::make(true);

  // Int types
  ExprHandle int_f1 = (a > b) && IntImm::make(1);
  ExprHandle int_f2 = (c <= d) || IntImm::make(1);

  // Short types
  ExprHandle short_f1 = (a > b) && ShortImm::make(1);
  ExprHandle short_f2 = (c <= d) || ShortImm::make(1);

  // Long types
  ExprHandle long_f1 = (a > b) && LongImm::make(1);
  ExprHandle long_f2 = (c <= d) || LongImm::make(1);

  // Char types
  ExprHandle char_f1 = (a > b) && CharImm::make(1);
  ExprHandle char_f2 = (c <= d) || CharImm::make(1);

  // Byte types
  ExprHandle byte_f1 = (a > b) && ByteImm::make(1);
  ExprHandle byte_f2 = (c <= d) || ByteImm::make(1);

  SimpleIRExprEval eval1(bool_f1);
  SimpleIRExprEval eval2(bool_f2);
  SimpleIRExprEval eval3(int_f1);
  SimpleIRExprEval eval4(int_f2);
  SimpleIRExprEval eval5(short_f1);
  SimpleIRExprEval eval6(short_f2);
  SimpleIRExprEval eval7(long_f1);
  SimpleIRExprEval eval8(long_f2);
  SimpleIRExprEval eval9(char_f1);
  SimpleIRExprEval eval10(char_f2);
  SimpleIRExprEval eval11(byte_f1);
  SimpleIRExprEval eval12(byte_f2);

  ASSERT_EQ(eval1.value<bool>(), true);
  ASSERT_EQ(eval2.value<bool>(), true);
  ASSERT_EQ(eval3.value<int>(), 1);
  ASSERT_EQ(eval4.value<int>(), 1);
  ASSERT_EQ(eval5.value<int16_t>(), 1);
  ASSERT_EQ(eval6.value<int16_t>(), 1);
  ASSERT_EQ(eval7.value<int64_t>(), 1);
  ASSERT_EQ(eval8.value<int64_t>(), 1);
  ASSERT_EQ(eval9.value<int8_t>(), 1);
  ASSERT_EQ(eval10.value<int8_t>(), 1);
  ASSERT_EQ(eval11.value<uint8_t>(), 1);
  ASSERT_EQ(eval12.value<uint8_t>(), 1);
}

TEST(Expr, BitwiseOps) {
  ExprHandle a(59);
  ExprHandle b(11);
  ExprHandle c(101);
  ExprHandle d(2);
  ExprHandle f = (((a ^ (b << 1)) & c) >> 2) | d;

  SimpleIRExprEval eval(f);
  ASSERT_EQ(eval.value<int>(), 11);
}

TEST(Expr, DynamicShapeAdd) {
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
    SimpleIREvaluator(s, {a, b, c, n})(aData, bData, cData, size);
    ExpectAllNear(cData, std::vector<float>(size, 3.0f), 1e-7);
  };
  testWithSize(1);
  testWithSize(16);
  testWithSize(37);
}

TEST(Expr, OutOfBounds) {
  ExprHandle N(10);
  ExprHandle start(0);
  ExprHandle stop(15);
  VarHandle i("i", kInt);

  BufHandle X("X", {N}, kInt);

  auto body = Store::make(X, {i}, i);
  auto stmt = For::make(i, start, stop, body);

  PaddedBuffer<int> data(20);

  EXPECT_ANY_THROW(SimpleIREvaluator(stmt, {X})(data));
}

TEST(Expr, OutOfBounds2d) {
  std::vector<std::pair<int, int>> size_options = {{10, 15}, {15, 10}};
  for (auto sizes : size_options) {
    ExprHandle N(sizes.first);
    ExprHandle M(sizes.second);
    ExprHandle start(0);
    ExprHandle stopInner(15);
    ExprHandle stopOuter(15);
    VarHandle i("i", kInt);
    VarHandle j("j", kInt);

    BufHandle X("X", {N, M}, kInt);

    auto body = Store::make(X, {i, j}, i);
    auto inner = For::make(j, start, stopInner, body);
    auto stmt = For::make(i, start, stopOuter, inner);

    PaddedBuffer<int> data(400);

    EXPECT_ANY_THROW(SimpleIREvaluator(stmt, {X})(data));
  }
}

TEST(Expr, OutOfBounds2dFlattenedIndex) {
  ExprHandle buf_size(149);
  ExprHandle start(0);
  ExprHandle stopInner(15);
  ExprHandle stopOuter(10);
  VarHandle i("i", kInt);
  VarHandle j("j", kInt);

  BufHandle X("X", {buf_size}, kInt);

  auto idx = Add::make(Mul::make(i, stopInner), j);
  auto body = Store::make(X, {idx}, i);
  auto inner = For::make(j, start, stopInner, body);
  auto stmt = For::make(i, start, stopOuter, inner);

  PaddedBuffer<int> data(400);

  EXPECT_ANY_THROW(SimpleIREvaluator(stmt, {X})(data));
}

void testCond01() {
  const int N = 16;
  PaddedBuffer<float> a_v(N);
  BufHandle a_buf("a", {N}, kFloat);
  VarHandle index = VarHandle("index", kInt);
  StmtPtr assign_x2 = a_buf.store({index}, cast<float>(index) * 2);
  StmtPtr assign_x3 = a_buf.store({index}, cast<float>(index) * 3);
  ExprHandle even_cond = CompareSelect::make(Mod::make(index, 2), 0, kEQ);
  StmtPtr assign = Cond::make(even_cond, assign_x2, assign_x3);
  StmtPtr for_stmt = For::make(index, 0, N, assign);
  SimpleIREvaluator(for_stmt, {a_buf})(a_v);

  PaddedBuffer<float> a_ref(N);
  for (const auto i : c10::irange(N)) {
    if (i % 2 == 0) {
      a_ref(i) = i * 2;
    } else {
      a_ref(i) = i * 3;
    }
  }
  ExpectAllNear(a_v, a_ref, 1e-5);
}

void testIfThenElse01() {
  ExprHandle v = ifThenElse(ExprHandle(1), ExprHandle(1.0f), ExprHandle(2.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "IfThenElse(1, 1.f, 2.f)");

  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 1.0f);
}

void testIfThenElse02() {
  ExprHandle v = ifThenElse(ExprHandle(0), ExprHandle(1.0f), ExprHandle(2.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "IfThenElse(0, 1.f, 2.f)");

  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 2.0f);
}

void testIfThenElse03() {
  ExprHandle v =
      ifThenElse(BoolImm::make(false), ExprHandle(1.0f), ExprHandle(2.0f));

  std::ostringstream oss;
  oss << v;
  ASSERT_EQ(oss.str(), "IfThenElse(0, 1.f, 2.f)");

  SimpleIRExprEval eval(v);
  ASSERT_EQ(eval.value<float>(), 2.0f);
}

void testStmtClone() {
  const int N = 16;

  BufHandle a_buf("a", {N}, kInt);
  VarHandle index = VarHandle("index", kInt);
  StmtPtr body = a_buf.store({index}, 5);
  StmtPtr loop = For::make(index, 0, N, body);

  StmtPtr cloned_loop = Stmt::clone(loop);
  std::vector<int> orig_loop_results(N);
  std::vector<int> cloned_loop_results(N);
  SimpleIREvaluator(loop, {a_buf})(orig_loop_results);
  SimpleIREvaluator(cloned_loop, {a_buf})(cloned_loop_results);

  assertAllEqual(orig_loop_results, 5);
  assertAllEqual(cloned_loop_results, 5);

  // Let's add another assign to the body in the cloned loop and verify that the
  // original statement hasn't changed while the cloned one has.
  StmtPtr body_addition = a_buf.store({index}, 33);
  BlockPtr cloned_body = static_to<Block>(static_to<For>(cloned_loop)->body());
  cloned_body->append_stmt(body_addition);

  std::vector<int> orig_loop_results_after_mutation(N);
  std::vector<int> cloned_loop_results_after_mutation(N);
  SimpleIREvaluator(loop, {a_buf})(orig_loop_results_after_mutation);
  SimpleIREvaluator(cloned_loop, {a_buf})(cloned_loop_results_after_mutation);

  assertAllEqual(orig_loop_results_after_mutation, 5);
  assertAllEqual(cloned_loop_results_after_mutation, 33);
}

} // namespace jit
} // namespace torch
